#!/usr/bin/env python3
"""Combine the disjoint reporting shards without selecting thresholds or checkpoints."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import summarize
from evaluate_training_loss_ablation import ARTIFACTS, CONDITIONS, MANIFEST, PREVIOUS, RESULTS, checkpoint_metadata


def main():
    output = RESULTS / "comparison.json"
    if output.exists():
        raise ValueError("comparison already exists")
    parts = [json.loads((RESULTS / f"part{i}.json").read_text()) for i in range(2)]
    previous = json.loads(PREVIOUS.read_text())
    powdermill = json.loads(MANIFEST.read_text())["powdermill"]
    checkpoints = checkpoint_metadata(previous)
    for i, part in enumerate(parts):
        if part["shard"] != i or part["shards"] != 2 or part["checkpoints"] != checkpoints:
            raise ValueError("mismatched checkpoint/shard metadata")
        for key in ("previous_sha256", "manifest_sha256", "smoothing", "threshold", "binary_postprocessing", "code_sha256", "versions"):
            if part[key] != parts[0][key]:
                raise ValueError(f"incompatible evaluation shards: {key}")
    if parts[0]["previous_sha256"] != digest(PREVIOUS) or parts[0]["manifest_sha256"] != digest(MANIFEST):
        raise ValueError("source protocol changed")
    for path, expected in parts[0]["code_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"code changed during evaluation: {path}")
    rows, summary = {}, {}
    for name in CONDITIONS:
        rows[name] = sorted([row for part in parts for row in part["per_recording"][name]], key=lambda r: r["name"])
        if [row["name"] for row in rows[name]] != sorted(powdermill["evaluation"]):
            raise ValueError("missing or duplicated reporting segments")
        if sum(row["seconds"] for row in rows[name]) != powdermill["evaluated_seconds"]:
            raise ValueError("reporting duration changed")
        summary[name] = summarize(rows[name], {"full": 8})["full"]
        summary[name]["counts_tp_fp_fn"] = np.sum([r["area"]["full"]["counts"][8] for r in rows[name]], axis=0).tolist()
    old_head = torch.load(previous["inference"]["checkpoint"], map_location="cpu", weights_only=True)["head"]
    new_head = torch.load(ARTIFACTS / "soft_tv.pt", map_location="cpu", weights_only=True)["head"]
    control_reproduction = dict(head_weights_bitwise_equal=all(torch.equal(old_head[k], new_head[k]) for k in old_head),
        maximum_absolute_weight_difference=max(float((old_head[k] - new_head[k]).abs().max()) for k in old_head),
        previous_summary=previous["summary"]["gaussian_probability"],
        metric_difference={k: summary["soft_tv"][k] - previous["summary"]["gaussian_probability"][k]
            for k in ("ap", "iou", "pooled_precision", "pooled_recall")})
    report = dict(protocol="protocol.json", protocol_sha256=digest(RESULTS / "protocol.json"),
        evaluation_shards={f"part{i}.json": digest(RESULTS / f"part{i}.json") for i in range(2)},
        checkpoints=checkpoints, summary=summary, control_reproduction=control_reproduction,
        per_recording=rows, reporting_segments=30, evaluated_seconds=powdermill["evaluated_seconds"],
        threshold=.08, threshold_selection="none; frozen from previous control calibration on Recordings 2–4",
        legacy_inference_reproduction="bitwise exact raw prediction cache match on every reporting segment",
        training_pairing="initial adapter weights, recording IDs, per-epoch sample order and CPU RNG hashes match",
        limitations="single training seed and one original reporting recording; fixed-threshold IoU includes calibration effects")
    write_json(output, report)
    labels = {"soft_tv": "Soft targets + TV (control)", "soft_no_tv": "Soft targets, no TV",
        "hard_tv": "Hard targets + TV", "hard_no_tv": "Hard targets, no TV (BCE only)"}
    with (RESULTS / "summary.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "mask_ap", "mean_2d_iou", "pooled_precision", "pooled_recall", "threshold"])
        for name, score in summary.items():
            writer.writerow([name, *[score[k] for k in ("ap", "iou", "pooled_precision", "pooled_recall", "threshold")]])
    lines = ["# Training-loss ablation: SongMAE-Large, 10k seconds", "",
        "All four models were retrained using the same self-reviewed Qwen labels: 2,103 windows from 266 XC recordings. "
        "Same seed (0), initial adapter, minibatch order, frozen backbone revision, AdamW and final checkpoint after three epochs. "
        "No training/validation split was introduced.", "",
        "Inference is fixed: probability-space Gaussian smoothing (σ = 2 mel bins, 3 frames), threshold 0.08; no other mask cleanup. "
        "The threshold comes from the previous control's calibration on original Powdermill Recordings 2–4; it was not reselected.", "",
        "Identical 8,235 seconds across 30 segments of original Recording_1. AP is exact pixel AP of continuous smoothed scores, "
        "averaged over 29 positive segments; IoU is averaged over all 30 (empty union = 1). Precision/recall pool pixel counts.", "",
        "| Training loss | Mask AP ↑ | 2D IoU ↑ | Precision | Recall |", "|---|---:|---:|---:|---:|"]
    for name, score in summary.items():
        lines.append(f"| {labels[name]} | {score['ap']:.3f} | {score['iou']:.3f} | {score['pooled_precision']:.3f} | {score['pooled_recall']:.3f} |")
    lines += ["", "These are single-seed development results, not significance estimates or unseen-test performance. "
        "Removing target smoothing can change confidence calibration, so fixed-threshold IoU and threshold-free AP answer different questions.", "",
        "The old checkpoint reproduced its saved raw predictions bit-for-bit on all 30 segments using the current inference path. "
        f"Rerun control head identical to the old head: {control_reproduction['head_weights_bitwise_equal']}; "
        f"AP difference: {control_reproduction['metric_difference']['ap']:+.8f}; "
        f"IoU difference: {control_reproduction['metric_difference']['iou']:+.8f}.", "",
        "`protocol.json` records the frozen design; `comparison.json` contains full precision results and paired-training checks; "
        "`part0.json`/`part1.json` include source/code hashes. Checkpoints and full-resolution probability caches are under "
        f"`{ARTIFACTS}`. The annotation pause snapshot is preserved there too."]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(summary=summary, control_reproduction=control_reproduction), indent=2))


if __name__ == "__main__":
    main()
