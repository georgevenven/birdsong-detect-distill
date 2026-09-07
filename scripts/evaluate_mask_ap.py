#!/usr/bin/env python3
import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.evaluation import WABAD_SITES, powdermill, wabad, xcsl
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import load_heads, reference_mask, songmae_probabilities


def ap(truth, probability, bins=4096):
    if not truth.any():
        return np.nan
    index = np.minimum((probability * (bins - 1)).astype(np.int16), bins - 1)
    positive = np.bincount(index[truth].ravel(), minlength=bins)[::-1]
    negative = np.bincount(index[~truth].ravel(), minlength=bins)[::-1]
    true = np.cumsum(positive)
    precision = true / np.maximum(true + np.cumsum(negative), 1)
    return float(np.sum(positive / positive.sum() * precision))


def main():
    parser = argparse.ArgumentParser(description="Evaluate recording-mean temporal and 2D foreground-mask AP.")
    parser.add_argument("--dataset", choices=("powdermill", "xcsl", "wabad"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    saved = {path: torch.load(path, map_location="cpu", weights_only=True) for path in args.checkpoint}
    grouped = {}
    for path, checkpoint in saved.items():
        grouped.setdefault(checkpoint["backbone_id"], []).append(path)
    source = wabad(args.root, WABAD_SITES) if args.dataset == "wabad" else {
        "powdermill": powdermill, "xcsl": xcsl}[args.dataset](args.root)
    source = list(islice(source, args.maximum) if args.maximum else source)
    scores = {path.stem: [] for path in args.checkpoint}
    device = torch.device(args.device)
    for backbone_id, paths in grouped.items():
        backbone = load_backbone(backbone_id, device)
        heads = load_heads(paths, backbone, device)
        for index, (_, waveform, events) in enumerate(source, 1):
            probabilities = songmae_probabilities(backbone, heads, waveform, device)
            truth = reference_mask(events, next(iter(probabilities.values())).shape)
            for name, probability in probabilities.items():
                scores[name].append((ap(truth.any(0), probability.max(0)), ap(truth, probability)))
            print(f"{backbone_id.split('/')[-1]}: {index}/{len(source)}", flush=True)
        del backbone, heads
        torch.cuda.empty_cache()
    models = {}
    for path in args.checkpoint:
        values = np.asarray(scores[path.stem])
        models[path.stem] = {"checkpoint": str(path), "backbone": saved[path]["backbone_id"],
            "temporal_ap": float(np.nanmean(values[:, 0])), "mask_ap_2d": float(np.nanmean(values[:, 1])),
            "positive_recordings": int(np.isfinite(values[:, 1]).sum())}
    result = {"dataset": args.dataset, "files": len(source),
        "metrics": ["temporal_ap", "mask_ap_2d"], "aggregation": "recording_mean",
        "prediction_representation": "raw_probability_union_mask", "primary_postprocessing": "none",
        "probability_bins": 4096, "coordinate_space": "200_hz_time_x_128_mel_frequency", "models": models}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"mask_ap_{args.dataset}_{len(source)}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
