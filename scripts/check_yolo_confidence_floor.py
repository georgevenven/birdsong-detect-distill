#!/usr/bin/env python3
"""Measure AP sensitivity to YOLO's candidate floor on development recordings only."""
import argparse
import json
from pathlib import Path

import numpy as np

from birdsong_detect_distill.baseline_models import Predictor, maps
from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import area_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Matched teacher-trained YOLO checkpoint")
    parser.add_argument("--cache", type=Path, required=True, help="Existing primary benchmark prediction cache")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--floors", type=float, nargs=2, default=(.00001, .0001), metavar=("LOW", "HIGH"))
    parser.add_argument("--all-development", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new output file")
    low, high = args.floors
    if not 0 < low < high <= .001:
        raise ValueError("require 0 < LOW < HIGH <= 0.001")
    manifest = json.loads(args.manifest.read_text())
    records = [r for r in recordings(args.root, "powdermill") if r["name"] in manifest["powdermill"]["calibration"]]
    if not args.all_development:
        records = [next(r for r in records if r["group"] == group) for group in sorted({r["group"] for r in records})]
    results = {}
    for kind in ("birdbox", "qwen_yolo"):
        predictor = Predictor(kind, args.checkpoint if kind == "qwen_yolo" else None, args.device, confidence=low, max_det=args.max_det)
        cached_floor = json.loads((args.cache / kind / "metadata.json").read_text())["confidence_floor"]
        rows = []
        for record in records:
            waveform, rate = read_audio(record)
            duration = len(waveform) / rate
            waveform, rate = model_audio(waveform, rate, kind)
            boxes = predictor.predict(waveform, rate)["boxes"]
            row = dict(recording=record["name"], boxes=len(boxes), ap={})
            path = args.cache / kind / "powdermill" / f"{record['name']}.npz"
            with np.load(path) as saved:
                row["primary_candidates_identical"] = np.array_equal(boxes[boxes[:, 4] >= cached_floor], saved["boxes"])
            for floor in (low, high):
                kept = boxes[boxes[:, 4] >= floor]
                probability, temporal = maps(dict(boxes=kept), duration)
                metrics = area_scores(probability, temporal, record["events"], len(temporal), [[0, duration]])
                row["ap"][str(floor)] = {k: v["ap"] for k, v in metrics.items()}
            row["ap_change"] = {k: row["ap"][str(low)][k] - row["ap"][str(high)][k]
                if row["ap"][str(low)][k] is not None else None for k in ("full", "common", "temporal")}
            rows.append(row)
            print(kind, record["name"], row["ap_change"], flush=True)
        results[kind] = dict(inference=predictor.metadata, recordings=rows,
            mean_ap_change={k: float(np.mean([r["ap_change"][k] for r in rows if r["ap_change"][k] is not None]))
                for k in ("full", "common", "temporal")})
        del predictor
    write_json(args.out, dict(manifest_sha256=digest(args.manifest), floors=[low, high],
        selection="all development" if args.all_development else "first segment of each calibration source; no reporting data",
        diagnostic_only=not args.all_development, results=results))


if __name__ == "__main__":
    main()
