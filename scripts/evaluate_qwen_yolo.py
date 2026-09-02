#!/usr/bin/env python3
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import batched, islice
from pathlib import Path

import librosa
import numpy as np
from ultralytics import YOLO

from birdsong_detect_distill.evaluation import WABAD_SITES, counters, metrics, powdermill, score_recording, wabad, xcsl
from birdsong_detect_distill.qwen import image


def spectrogram(waveform):
    value = librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=1024,
        hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000)
    return librosa.power_to_db(value, ref=np.max, top_db=None).astype(np.float32)


def windows(waveform):
    spec, width, hop = spectrogram(waveform), 1000, 500
    starts = list(range(0, max(1, spec.shape[1] - width + hop), hop))
    if starts[-1] + width < spec.shape[1]:
        starts.append(starts[-1] + hop)
    output = []
    for start in starts:
        valid = min(width, spec.shape[1] - start)
        window = np.pad(spec[:, start:start + valid], ((0, 0), (0, width - valid)), constant_values=-100)
        output.append((start, valid, np.ascontiguousarray(np.asarray(image(window))[:, :, ::-1])))
    return spec.shape[1], output


def predict_group(model, waveforms, batch_size, device, workers):
    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(windows, waveforms))
    jobs = [(index, *window) for index, (_, values) in enumerate(records) for window in values]
    results = []
    for chunk in batched([x[3] for x in jobs], batch_size):
        results.extend(model.predict(list(chunk), imgsz=1024, device=device, conf=.001,
            iou=.7, batch=len(chunk), verbose=False))
    scores = [np.zeros(length, np.float32) for length, _ in records]
    for result, (index, start, valid, _) in zip(results, jobs):
        for box, confidence in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            left, right = max(0, int(np.floor(box[0] * 1000))), min(valid, int(np.ceil(box[2] * 1000)))
            scores[index][start + left:start + right] = np.maximum(scores[index][start + left:start + right], confidence)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Qwen-distilled YOLO detector on BirdCODE datasets.")
    parser.add_argument("--dataset", choices=("wabad", "powdermill", "xcsl"), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/birdcode/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/yolo11n-qwen-birdbox-init.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--name", default="birdbox_qwen")
    parser.add_argument("--initialization", default="released_birdbox_yolo11n")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recording-batch-size", type=int, default=4)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    model, frame, event, sites = YOLO(args.checkpoint), *counters(), {}
    loader = {"wabad": wabad, "powdermill": powdermill, "xcsl": xcsl}[args.dataset]
    source = wabad(args.root, WABAD_SITES[args.shard_index::args.shards]) if args.dataset == "wabad" else loader(args.root)
    source = islice(source, args.max_files) if args.max_files else source
    files = 0
    for group in batched(source, args.recording_batch_size):
        scores = predict_group(model, [x[1] for x in group], args.batch_size, args.device, args.preprocess_workers)
        for (recording, waveform, annotations), score in zip(group, scores):
            files += 1
            reference = np.asarray(annotations).reshape(-1, 4)[:, :2]
            if args.dataset == "wabad":
                site = recording.split("/", 1)[0]
                if site not in sites:
                    sites[site] = counters()
                score_recording(score, 200, reference, len(waveform) / 32000, *sites[site])
            else:
                score_recording(score, 200, reference, len(waveform) / 32000, frame, event)
        print(f"{args.dataset}: {files} files", flush=True)
    if sites:
        frame = sum((x[0] for x in sites.values()), np.zeros_like(frame))
        event = {iou: sum((x[1][iou] for x in sites.values()), np.zeros_like(event[iou])) for iou in event}
    result = {"model": str(args.checkpoint), "initialization": args.initialization,
        "fine_tuned": args.initialization != "random_yolo11n",
        "teacher": "Qwen3.8-27B-Q8_0", "dataset": args.dataset, "task": "binary_any_bird_detection", "files": files,
        **metrics(frame, event)}
    if sites:
        per_site = {site: metrics(*value) for site, value in sites.items()}
        result.update(site_macro={key: float(np.mean([x[key] for x in per_site.values()]))
            for key in next(iter(per_site.values()))}, sites=len(sites), per_site=per_site)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.shard_index}of{args.shards}" if args.shards > 1 else ""
    path = args.out / f"{args.name}_{args.dataset}{suffix}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(path.with_suffix(".npz"), frame=frame, event_02=event[.2], event_05=event[.5])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
