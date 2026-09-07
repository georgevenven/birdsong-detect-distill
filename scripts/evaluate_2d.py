#!/usr/bin/env python3
import argparse
import json
from itertools import batched, islice
from pathlib import Path

import librosa
import numpy as np
import torch

from birdsong_detect_distill.evaluation import WABAD_SITES, powdermill, wabad, xcsl
from birdsong_detect_distill.model import DenseHead, load_backbone


THRESHOLDS = np.linspace(0, 1, 101)


def references(events):
    events = np.asarray(events).reshape(-1, 4)
    low = librosa.hz_to_mel(20)
    scale = 128 / (librosa.hz_to_mel(16000) - low)
    return np.column_stack((events[:, 0], (librosa.hz_to_mel(np.clip(events[:, 2], 20, 16000)) - low) * scale,
        events[:, 1], (librosa.hz_to_mel(np.clip(events[:, 3], 20, 16000)) - low) * scale))


def reference_mask(events, shape):
    mask = np.zeros(shape, bool)
    for left, low, right, high in references(events):
        x0, x1 = max(0, int(np.floor(left * 200))), min(shape[1], int(np.ceil(right * 200)))
        y0, y1 = max(0, int(np.floor(low))), min(shape[0], int(np.ceil(high)))
        mask[y0:y1, x0:x1] = True
    return mask


def iou_curve(probability, truth):
    bins = np.minimum((probability * (len(THRESHOLDS) - 1)).astype(np.int16), len(THRESHOLDS) - 1)
    positive = np.bincount(bins[truth], minlength=len(THRESHOLDS))
    negative = np.bincount(bins[~truth], minlength=len(THRESHOLDS))
    intersection = np.cumsum(positive[::-1])[::-1]
    union = int(truth.sum()) + np.cumsum(negative[::-1])[::-1]
    return np.divide(intersection, union, out=np.ones(len(THRESHOLDS)), where=union != 0)


def recording_curves(probability, truth):
    return np.stack((iou_curve(probability.max(0), truth.any(0)), iou_curve(probability, truth)))


def summarize(curves, threshold):
    mean = np.mean(curves, axis=0)
    index = int(np.argmax(mean[1])) if threshold is None else int(np.argmin(np.abs(THRESHOLDS - threshold)))
    return {"temporal_iou": float(mean[0, index]), "iou_2d": float(mean[1, index]),
        "threshold": float(THRESHOLDS[index]), "recordings": len(curves),
        "threshold_selection": "maximum_recording_mean_2d_iou_on_development_set" if threshold is None else "fixed"}


def load_heads(paths, backbone, device):
    heads = {}
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = backbone.config
        head = DenseHead(config.enc_hidden_d, saved["hidden"], saved["height"], saved["width"],
            config.patch_height, config.patch_width, saved["dropout"]).to(device)
        head.load_state_dict(saved["head"])
        heads[path.stem] = head.eval()
    return heads


@torch.inference_mode()
def songmae_probabilities(backbone, heads, waveform, device):
    config = backbone.config
    spec = librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=1024,
        hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000)
    spec = (librosa.power_to_db(spec, ref=np.max, top_db=None).astype(np.float32) - config.audio_mean) / config.audio_std
    width, hop = 1000, 500
    starts = list(range(0, max(1, spec.shape[1] - width + hop), hop))
    if starts[-1] + width < spec.shape[1]:
        starts.append(starts[-1] + hop)
    output = {name: np.zeros((128, spec.shape[1]), np.float32) for name in heads}
    for indexes in batched(range(len(starts)), 4):
        indexes = list(indexes)
        valid = [min(width, spec.shape[1] - starts[i]) for i in indexes]
        values = np.asarray([np.pad(spec[:, starts[i]:starts[i] + valid[j]], ((0, 0), (0, width - valid[j])))
            for j, i in enumerate(indexes)])
        values = torch.from_numpy(values)[:, None].to(device)
        lengths = torch.tensor(valid, device=device)
        with torch.autocast(device.type):
            tokens = backbone(input_values=values, valid_timebins=lengths).last_hidden_state
            predictions = {name: head(tokens, lengths).sigmoid().float().cpu().numpy() for name, head in heads.items()}
        for name, prediction in predictions.items():
            for value, i, size in zip(prediction, indexes, valid):
                start = starts[i]
                output[name][:, start:start + size] = np.maximum(output[name][:, start:start + size], value[:, :size])
    return output


@torch.inference_mode()
def songmae_probability(backbone, head, waveform, device):
    return songmae_probabilities(backbone, {"model": head}, waveform, device)["model"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate recording-mean temporal and 2D IoU.")
    parser.add_argument("--dataset", choices=("powdermill", "xcsl", "wabad"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--out", type=Path, default=Path("results/reproduced"))
    args = parser.parse_args()
    checkpoints = args.checkpoint or [Path("artifacts/songmae-large-32x1-detector.pt")]
    saved = {path: torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints}
    grouped = {}
    for path, checkpoint in saved.items():
        grouped.setdefault(checkpoint["backbone_id"], []).append(path)
    if args.dataset == "wabad":
        maximum = args.maximum or len(WABAD_SITES)
        stride = max(1, len(WABAD_SITES) // maximum)
        source = (row for site in WABAD_SITES[::stride][:maximum] for row in islice(wabad(args.root, [site]), 1))
    else:
        source = {"powdermill": powdermill, "xcsl": xcsl}[args.dataset](args.root)
        source = islice(source, args.maximum) if args.maximum else source
    source = list(source)
    curves = {path.stem: [] for path in checkpoints}
    device = torch.device(args.device)
    for backbone_id, paths in grouped.items():
        backbone = load_backbone(backbone_id, device)
        heads = load_heads(paths, backbone, device)
        for index, (_, waveform, events) in enumerate(source, 1):
            probabilities = songmae_probabilities(backbone, heads, waveform, device)
            truth = reference_mask(events, next(iter(probabilities.values())).shape)
            for name, probability in probabilities.items():
                curves[name].append(recording_curves(probability, truth))
            print(f"{backbone_id.split('/')[-1]}: {index}/{len(source)}", flush=True)
        del backbone, heads
        torch.cuda.empty_cache()
    by_name = {path.stem: (path, saved[path]) for path in checkpoints}
    models = {name: {"checkpoint": str(path), "backbone": checkpoint["backbone_id"],
        "training": checkpoint.get("metrics", {}), **summarize(np.asarray(curves[name]), args.threshold)}
        for name, (path, checkpoint) in by_name.items()}
    result = {"dataset": args.dataset, "files": len(source), "metrics": ["temporal_iou", "iou_2d"],
        "aggregation": "recording_mean", "primary_postprocessing": "none",
        "empty_union_iou": 1.0,
        "coordinate_space": "200_hz_time_x_128_mel_frequency", "thresholds": len(THRESHOLDS),
        "reference_boxes": int(sum(len(row[2]) for row in source)), "models": models}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"iou_{args.dataset}_{len(source)}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
