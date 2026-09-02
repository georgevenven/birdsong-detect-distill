#!/usr/bin/env python3
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import batched, islice
from pathlib import Path

import numpy as np
import requests
from matplotlib import colormaps
from scipy import interpolate, signal
from ultralytics import YOLO

from birdsong_detect_distill.evaluation import WABAD_SITES, counters, metrics, powdermill, score_recording, wabad, xcsl


MODELS = {
    "yolo11n": ("87abf5c37f225f4a24df2a18c9bd64e6caf8c464a3730d413221909bf1599051", 2.6),
    "yolo11l": ("8806e5d798acc3f61b435e18114da284aa8fb0dfdc6abe2d71b349d1da024e17", 25.3),
}
URL = "https://raw.githubusercontent.com/org-arl/birdwatch-public/master/birdbox/models/{model}.pt"
MAGMA = (colormaps["magma"](np.arange(256))[:, :3][:, ::-1] * 255).astype(np.uint8)


def model_path(name, directory):
    path = directory / f"{name}.pt"
    expected = MODELS[name][0]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(URL.format(model=name), timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {path}")
    return path


def spectrogram_image(waveform, sample_rate=32000, duration=6):
    nfft = round(sample_rate / (44100 / 4096))
    hop = int(np.floor(sample_rate * duration / 1024)) + 1
    waveform = np.pad(waveform, nfft // 2)
    frequency, _, power = signal.spectrogram(waveform, sample_rate, signal.windows.hann(nfft, sym=True),
        nfft, nfft - hop, nfft, detrend=False, scaling="density", mode="psd")
    power = 10 * np.log10(np.maximum(power / max(power.max(), np.finfo(float).eps), 1e-8))
    power = power[(frequency >= 500) & (frequency <= 12000)]
    power = interpolate.interp1d(np.linspace(0, 1, len(power)), power, axis=0)(np.linspace(0, 1, 1024))
    low, high = np.percentile(power, (1, 99.8))
    value = np.clip((np.clip(power, low, high) - low) / max(high - low, np.finfo(float).eps), 0, 1) ** .85
    return np.ascontiguousarray(MAGMA[np.minimum((value[::-1] * 256).astype(np.int16), 255)])


def windows(waveform, sample_rate=32000, duration=6, overlap=1):
    length, stride = round(duration * sample_rate), duration - overlap
    for start in np.arange(0, max(0, len(waveform) / sample_rate - 1e-9) + 1e-12, stride):
        index = round(start * sample_rate)
        clip = np.pad(waveform[index:index + length], (0, max(0, index + length - len(waveform))))
        yield float(start), clip


def predict_group(model, waveforms, batch_size, device, confidence, workers):
    jobs = [(index, start, clip) for index, waveform in enumerate(waveforms) for start, clip in windows(waveform)]
    with ThreadPoolExecutor(workers) as pool:
        images = list(pool.map(lambda job: spectrogram_image(job[2]), jobs))
    owners = [(index, start) for index, start, _ in jobs]
    scores = [np.zeros(int(np.ceil(len(x) / 32000 * 100)), np.float32) for x in waveforms]
    results = []
    for chunk in batched(images, batch_size):
        results.extend(model.predict(list(chunk), imgsz=1024, device=device, conf=confidence, iou=.7,
            batch=len(chunk), verbose=False))
    for result, (index, start) in zip(results, owners):
        for box, score in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            left = max(0, int(np.floor((start + box[0] * 6) * 100)))
            right = min(len(scores[index]), int(np.ceil((start + box[2] * 6) * 100)))
            scores[index][left:right] = np.maximum(scores[index][left:right], score)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate released BirdBox weights without fine-tuning.")
    parser.add_argument("--dataset", choices=("wabad", "powdermill", "xcsl"), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/birdcode/raw"))
    parser.add_argument("--model", choices=MODELS, default="yolo11n")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recording-batch-size", type=int, default=8)
    parser.add_argument("--preprocess-workers", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=.15)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    model = YOLO(model_path(args.model, args.model_dir))
    frame, event, sites = *counters(), {}
    loader = {"wabad": wabad, "powdermill": powdermill, "xcsl": xcsl}[args.dataset]
    source = wabad(args.root, WABAD_SITES[args.shard_index::args.shards]) if args.dataset == "wabad" else loader(args.root)
    source = islice(source, args.max_files) if args.max_files else source
    files = 0
    for group in batched(source, args.recording_batch_size):
        scores = predict_group(model, [row[1] for row in group], args.batch_size, args.device,
            args.confidence, args.preprocess_workers)
        for (recording, waveform, annotations), score in zip(group, scores):
            files += 1
            reference = np.asarray(annotations).reshape(-1, 4)[:, :2]
            if args.dataset == "wabad":
                site = recording.split("/", 1)[0]
                if site not in sites:
                    sites[site] = counters()
                score_recording(score, 100, reference, len(waveform) / 32000, *sites[site])
            else:
                score_recording(score, 100, reference, len(waveform) / 32000, frame, event)
        print(f"{args.dataset}: {files} files", flush=True)
    if sites:
        frame = sum((value[0] for value in sites.values()), np.zeros_like(frame))
        event = {iou: sum((value[1][iou] for value in sites.values()), np.zeros_like(event[iou])) for iou in event}
    result = {"model": f"org-arl/birdwatch-public:{args.model}.pt", "fine_tuned": False,
        "dataset": args.dataset, "task": "binary_any_bird_detection", "files": files, "confidence_floor": args.confidence,
        **metrics(frame, event)}
    if sites:
        per_site = {site: metrics(*value) for site, value in sites.items()}
        result.update(site_macro={key: float(np.mean([x[key] for x in per_site.values()]))
            for key in next(iter(per_site.values()))}, sites=len(sites), per_site=per_site)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.shard_index}of{args.shards}" if args.shards > 1 else ""
    path = args.out / f"birdbox_{args.model}_{args.dataset}{suffix}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(path.with_suffix(".npz"), frame=frame, event_02=event[.2], event_05=event[.5])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
