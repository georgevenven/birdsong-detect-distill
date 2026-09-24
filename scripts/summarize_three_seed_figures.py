#!/usr/bin/env python3
"""Freeze seed-zero references, then summarize fixed-data training-seed variability."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from evaluate_current_backbones import (ANCHOR, RESULTS, SCALING_LABELS, SCALING_RESULTS,
    MULTISEED_RESULTS, MULTISEED_RUN)


CONFIGS = ("large_s100", "large_s1000", "large_s10000", "micro_s10000", "base_s10000")


def unpack(path):
    report = json.loads(path.read_text())
    if path == ANCHOR:
        return (report["protocol"]["checkpoints"]["self_review"], report["per_recording"]["student_self_review"],
            report["summary"]["student_self_review"], report["protocol"])
    return report["protocol"]["checkpoint"], report["per_recording"], report["summary"], report["protocol"]


def freeze():
    paths = {"large_s100": SCALING_RESULTS / "s100/comparison.json",
        "large_s1000": SCALING_RESULTS / "s1000/comparison.json", "large_s10000": ANCHOR,
        "micro_s10000": RESULTS / "micro/comparison.json", "base_s10000": RESULTS / "base/comparison.json"}
    references, protected = {}, {}
    anchor = json.loads(ANCHOR.read_text())["protocol"]
    for name, path in paths.items():
        saved, rows, score, protocol = unpack(path)
        if (saved["seed"] != 0 or saved["target_smoothing"] or saved["tv_weight"] != 0
                or protocol["partitions"] != anchor["partitions"]
                or summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score
                or digest(saved["checkpoint"]) != saved["checkpoint_sha256"]):
            raise ValueError(f"invalid seed-zero reference: {name}")
        references[name] = dict(report=str(path), report_sha256=digest(path),
            checkpoint=saved["checkpoint"], checkpoint_sha256=saved["checkpoint_sha256"])
        protected[str(path)] = digest(path)
        protected[saved["checkpoint"]] = saved["checkpoint_sha256"]
    split = json.loads(Path(anchor["split"]).read_text())
    paths_to_protect = [Path(anchor["split"]), SCALING_LABELS / "manifest.json"]
    paths_to_protect.extend(Path(part["files"]["self_review"]["path"]) for part in split["partitions"].values())
    paths_to_protect.extend(SCALING_LABELS.glob("*.jsonl"))
    paths_to_protect.extend(SCALING_RESULTS.glob("xc_scaling_and_backbones.*"))
    protected.update({str(path): digest(path) for path in paths_to_protect})
    plan = dict(seeds=[0, 1, 2], configurations=list(CONFIGS), seed_zero=references,
        protected_files=protected, partitions=anchor["partitions"],
        data_policy="fixed training subsets, XC validation, teacher annotations and Powdermill recordings across seeds",
        varying="adapter initialization, stochastic training order and dropout via the training seed",
        checkpoint_selection="per run: minimum XC validation BCE over three epochs",
        threshold_selection=anchor["threshold_selection"],
        error_bars="mean plus/minus one sample standard deviation across three training seeds (ddof=1), not a confidence interval")
    path = MULTISEED_RESULTS / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen three-seed experiment or seed-zero files changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.prepare:
        print("Verified five reusable seed-zero runs; ten new training/evaluation runs planned.")
        return
    if (MULTISEED_RESULTS / "summary.json").exists():
        raise ValueError("completed three-seed summary exists; preserve it")
    models, sources, table = {}, {}, []
    for name in CONFIGS:
        reference_path = Path(plan["seed_zero"][name]["report"])
        first, reference_rows, _, _ = unpack(reference_path)
        size, seconds = name.split("_s")
        values, hashes = [], {}
        for seed in plan["seeds"]:
            path = reference_path if seed == 0 else MULTISEED_RESULTS / f"seed{seed}" / name / "comparison.json"
            saved, rows, score, protocol = unpack(path)
            for key in ("annotations_sha256", "validation_annotations_sha256", "training_recording_ids",
                    "validation_recording_ids", "training_config", "backbone_id", "backbone_revision",
                    "target_smoothing", "tv_weight", "hidden", "height", "width", "dropout"):
                if saved[key] != first[key]:
                    raise ValueError(f"unmatched {key}: {name}, seed {seed}")
            for key in ("epochs", "train_timebins", "validation_timebins", "train_windows", "validation_windows",
                    "checkpoint_selection", "threshold_source"):
                if saved["metrics"][key] != first["metrics"][key]:
                    raise ValueError(f"unmatched training/selection procedure: {name}, seed {seed}")
            if (saved["seed"] != seed or protocol["partitions"] != plan["partitions"]
                    or digest(saved["checkpoint"]) != saved["checkpoint_sha256"]):
                raise ValueError(f"changed seed/checkpoint/evaluation partition: {name}, {seed}")
            actual = torch.load(saved["checkpoint"], map_location="cpu", weights_only=True)
            if actual["seed"] != seed or actual["annotations_sha256"] != first["annotations_sha256"]:
                raise ValueError("checkpoint metadata differs from its report")
            if saved["metrics"]["selected_epoch"] != min(saved["training_history"], key=lambda r: r["validation_loss"])["epoch"]:
                raise ValueError("checkpoint was not selected by validation BCE")
            for partition, expected in reference_rows.items():
                if len(rows[partition]) != len(expected):
                    raise ValueError("incomplete reporting coverage")
                for row, old in zip(rows[partition], expected):
                    if any(row[k] != old[k] for k in ("name", "group", "seconds")):
                        raise ValueError("different evaluation intervals")
                    tp, _, fn = row["area"]["full"]["counts"][0]
                    old_tp, _, old_fn = old["area"]["full"]["counts"][0]
                    if tp + fn != old_tp + old_fn:
                        raise ValueError("different reference mask")
            if summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score:
                raise ValueError("seed metrics do not reproduce")
            value = dict(seed=seed, **score, checkpoint=saved["checkpoint"],
                checkpoint_sha256=saved["checkpoint_sha256"], selected_epoch=saved["metrics"]["selected_epoch"])
            values.append(value)
            hashes[str(seed)] = saved["checkpoint_sha256"]
            sources[str(path)] = digest(path)
            table.append([name, seed, score["ap"], score["iou"], score["threshold"], value["selected_epoch"]])
        models[name] = dict(training_seconds=int(seconds), seeds=plan["seeds"], per_seed=values,
            checkpoint_sha256_by_seed=hashes,
            mask_ap_2d=float(np.mean([row["ap"] for row in values])),
            mask_ap_2d_std=float(np.std([row["ap"] for row in values], ddof=1)),
            iou_2d=float(np.mean([row["iou"] for row in values])),
            iou_2d_std=float(np.std([row["iou"] for row in values], ddof=1)))
    anchor = json.loads(ANCHOR.read_text())
    shared = dict(dataset="powdermill", evaluated_seconds=10800, segments=36, source_recordings=["Recording_1"],
        aggregation="segment_mean_then_seed_mean", seeds=plan["seeds"], validation_seconds=800,
        error_bars=plan["error_bars"], sources_sha256=sources,
        qwen_self_review=dict(mask_ap_2d=anchor["summary"]["teacher_self_review"]["ap"]))
    write_json(MULTISEED_RESULTS / "scaling.json", dict(**shared,
        students=[models[f"large_s{seconds}"] for seconds in (100, 1000, 10000)]))
    write_json(MULTISEED_RESULTS / "backbones.json", dict(**shared, training_seconds=10000,
        models={size: models[f"{size}_s10000"] for size in ("micro", "base", "large")}))
    write_json(MULTISEED_RESULTS / "summary.json", dict(**shared, configurations=models,
        manifest_sha256=digest(MULTISEED_RESULTS / "manifest.json")))
    with (MULTISEED_RESULTS / "per_seed.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["Configuration", "Training seed", "Mask AP", "2D IoU", "Threshold", "Selected epoch"])
        writer.writerows(table)
    (MULTISEED_RESULTS / "caption.txt").write_text(
        "Figure 3. Effect of (a) training-label budget for SongMAE-Large and (b) backbone size at 10,000 s of training audio. "
        "Points/bars show mean mask AP; error bars show ±1 sample standard deviation over three training seeds (0, 1, 2), "
        "with fixed data subsets and self-reviewed Qwen labels. All runs use the same separate 800 s XC validation set for "
        "checkpoint selection and the same Powdermill recordings for calibration and evaluation. AP averages the 35 positive "
        "segments of complete Recording_1. The dashed line denotes the fixed Qwen teacher. Error bars describe training-seed "
        "variability, not uncertainty from dataset sampling.\n")
    print(json.dumps({name: {k: value[k] for k in ("mask_ap_2d", "mask_ap_2d_std", "iou_2d", "iou_2d_std")}
        for name, value in models.items()}, indent=2))


if __name__ == "__main__":
    main()
