#!/usr/bin/env python3
import argparse
import json
from itertools import batched, islice
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from birdsong_detect_distill.evaluation import WABAD_SITES, counters, metrics, powdermill, score_recording, wabad, xcsl
from birdsong_detect_distill.model import clean_mask, load_detector


def spectrogram(waveform, config):
    value = librosa.feature.melspectrogram(y=waveform, sr=config.audio_sr, n_fft=config.audio_fft,
        hop_length=config.audio_hop_size, power=2, n_mels=config.mels, fmin=20, fmax=config.audio_sr // 2)
    value = librosa.power_to_db(value, ref=np.max, top_db=None).astype(np.float32)
    return (value - config.audio_mean) / config.audio_std


@torch.inference_mode()
def predict_many(waveforms, backbone, head, batch_size, device):
    config = backbone.config
    windows, lengths, widths = [], [], []
    for waveform in waveforms:
        spec, width = spectrogram(waveform, config), config.num_timebins
        starts = list(range(0, max(1, spec.shape[1] - width + width // 2), width // 2))
        if starts[-1] + width < spec.shape[1]:
            starts.append(starts[-1] + width // 2)
        valid = [min(width, spec.shape[1] - start) for start in starts]
        for start, size in zip(starts, valid):
            window = np.zeros((config.mels, width), np.float32)
            window[:, :size] = spec[:, start:start + size]
            windows.append(window)
        lengths.append(valid)
        widths.append(spec.shape[1])
    output = []
    valid = [x for record in lengths for x in record]
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(windows)).unsqueeze(1), torch.tensor(valid)), batch_size)
    for batch, sizes in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            tokens = backbone(input_values=batch.to(device), valid_timebins=sizes.to(device)).last_hidden_state
            probability = head(tokens, sizes).sigmoid()
        output.extend(probability.float().cpu().numpy())
    scores, offset = [], 0
    for record_valid, bins in zip(lengths, widths):
        record = output[offset:offset + len(record_valid)]
        kept = [record[0][:, :record_valid[0]]] if len(record) == 1 else [value[:, :750] if index == 0
            else value[:, 250:(record_valid[index] if index == len(record) - 1 else 750)] for index, value in enumerate(record)]
        smooth, _ = clean_mask(np.concatenate(kept, 1)[:, :bins])
        scores.append((smooth.max(0), config.audio_sr / config.audio_hop_size))
        offset += len(record_valid)
    return scores


def main():
    parser = argparse.ArgumentParser(description="BirdCODE-protocol binary bird-detection evaluation.")
    parser.add_argument("--dataset", choices=("wabad", "powdermill", "xcsl"), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/birdcode/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/songmae-large-32x1-detector.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--recording-batch-size", type=int, default=1)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, head, saved = load_detector(args.checkpoint, device)
    frame, event = counters()
    sites = {}
    loader = {"wabad": wabad, "powdermill": powdermill, "xcsl": xcsl}[args.dataset]
    source = wabad(args.root, WABAD_SITES[args.shard_index::args.shards]) if args.dataset == "wabad" else loader(args.root)
    source = islice(source, args.max_files) if args.max_files else source
    files = 0
    for group in batched(source, args.recording_batch_size):
        predictions = predict_many([row[1] for row in group], backbone, head, args.batch_size, device)
        for (recording, waveform, annotations), (probability, rate) in zip(group, predictions):
            files += 1
            reference = np.asarray(annotations).reshape(-1, 4)[:, :2]
            if args.dataset == "wabad":
                site = recording.split("/", 1)[0]
                if site not in sites:
                    sites[site] = counters()
                score_recording(probability, rate, reference, len(waveform) / 32000, *sites[site])
            else:
                score_recording(probability, rate, reference, len(waveform) / 32000, frame, event)
            if files % 10 == 0:
                print(f"{args.dataset}: {files} files", flush=True)
    if sites:
        frame = sum((value[0] for value in sites.values()), np.zeros_like(frame))
        event = {iou: sum((value[1][iou] for value in sites.values()), np.zeros_like(event[iou])) for iou in event}
    result = {"model": saved["backbone_id"], "dataset": args.dataset, "task": "binary_any_bird_detection",
        "files": files, **metrics(frame, event)}
    if sites:
        per_site = {site: metrics(*value) for site, value in sites.items()}
        result["site_macro"] = {key: float(np.mean([value[key] for value in per_site.values()])) for key in next(iter(per_site.values()))}
        result["sites"] = len(sites)
        result["per_site"] = per_site
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.shard_index}of{args.shards}" if args.shards > 1 else ""
    path = args.out / f"songmae_large_32x1_{args.dataset}{suffix}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(path.with_suffix(".npz"), frame=frame, event_02=event[.2], event_05=event[.5])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
