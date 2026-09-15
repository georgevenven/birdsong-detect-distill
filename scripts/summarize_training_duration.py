#!/usr/bin/env python3
"""Matched 3-versus-15-epoch, fixed-10k-label experiment on Powdermill only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from evaluate_current_backbones import ANCHOR, DURATION_RESULTS, MULTISEED_RESULTS, RESULTS
from summarize_three_seed_figures import unpack


SIZES, SEEDS = ("micro", "base", "large"), (0, 1, 2)


def freeze():
    anchor = json.loads(ANCHOR.read_text())["protocol"]
    baseline_summary = json.loads((MULTISEED_RESULTS / "summary.json").read_text())
    protected = {str(path): digest(path) for path in MULTISEED_RESULTS.glob("xc_scaling_and_backbones.*")}
    protected[str(MULTISEED_RESULTS / "summary.json")] = digest(MULTISEED_RESULTS / "summary.json")
    for path, sha in anchor["code_sha256"].items():
        if path in ("scripts/train.py", "src/birdsong_detect_distill/model.py", "src/birdsong_detect_distill/data.py"):
            if digest(path) != sha:
                raise ValueError(f"training implementation changed: {path}")
            protected[path] = sha
    split = json.loads(Path(anchor["split"]).read_text())
    protected[anchor["split"]] = anchor["split_sha256"]
    for part in split["partitions"].values():
        labels = part["files"]["self_review"]
        protected[labels["path"]] = labels["sha256"]
    baselines = {}
    for size in SIZES:
        for seed in SEEDS:
            path = (ANCHOR if size == "large" else RESULTS / size / "comparison.json") if seed == 0 else (
                MULTISEED_RESULTS / f"seed{seed}" / f"{size}_s10000" / "comparison.json")
            saved, rows, score, protocol = unpack(path)
            listed = baseline_summary["configurations"][f"{size}_s10000"]["per_seed"][seed]
            if (saved["seed"] != seed or saved["metrics"]["epochs"] != 3
                    or saved["metrics"]["train_timebins"] != 2000000
                    or saved["metrics"]["validation_timebins"] != 160000
                    or saved["target_smoothing"] or saved["tv_weight"] != 0
                    or protocol["partitions"] != anchor["partitions"]
                    or summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score
                    or listed["ap"] != score["ap"] or listed["checkpoint_sha256"] != saved["checkpoint_sha256"]):
                raise ValueError(f"invalid three-epoch reference: {size}, seed {seed}")
            baselines[f"{size}_seed{seed}"] = dict(report=str(path), report_sha256=digest(path),
                checkpoint=saved["checkpoint"], checkpoint_sha256=saved["checkpoint_sha256"])
            protected[str(path)] = digest(path)
            protected[saved["checkpoint"]] = saved["checkpoint_sha256"]
    for path, sha in protected.items():
        if digest(path) != sha:
            raise ValueError(f"protected input changed: {path}")
    code = ("scripts/run_training_duration.sh", "scripts/summarize_training_duration.py",
        "scripts/evaluate_current_backbones.py", "scripts/evaluate_2d.py", "scripts/evaluate_songmae_smoothing.py",
        "src/birdsong_detect_distill/benchmark_data.py", "src/birdsong_detect_distill/benchmark_metrics.py")
    plan = dict(dataset="powdermill", training_seconds=10000, validation_seconds=800, sizes=list(SIZES),
        seeds=list(SEEDS), epochs=[3, 15], baselines=baselines, protected_files=protected,
        code_sha256={path: digest(path) for path in code}, partitions=anchor["partitions"],
        varied="maximum training epochs only; nine fresh runs using unchanged train.py and fixed data",
        checkpoint_selection="minimum XC validation BCE within each epoch budget, never Powdermill AP",
        threshold_selection=anchor["threshold_selection"],
        inference="unchanged probability smoothing (2 mel bins, 15 ms), one calibrated threshold, no morphology",
        error_bars="mean plus/minus one sample SD across three fixed-data training seeds, not confidence intervals")
    path = DURATION_RESULTS / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen duration experiment changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def plot(models, teacher):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update({"font.size": 10.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.19, right=.98, bottom=.22, top=.96)
    x = np.arange(3)
    for epochs, offset, color in ((3, -.18, ".65"), (15, .18, "C0")):
        scores = [models[size][str(epochs)] for size in SIZES]
        axis.bar(x + offset, [row["ap_mean"] for row in scores], width=.33,
            yerr=[row["ap_std"] for row in scores], capsize=3, color=color, label=f"Up to {epochs} epochs")
    axis.axhline(teacher, color=".25", linestyle="--", linewidth=1.2, label="Qwen_teacher labels")
    axis.set(xticks=x, xticklabels=[s.title() for s in SIZES], ylim=(0, 1),
        ylabel="Powdermill mask AP", xlabel="Backbone (10,000 s training)")
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(.58, .025, "Mean ± SD (3 training seeds)", ha="center", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(DURATION_RESULTS / f"training_duration_ap.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.prepare:
        print("Verified nine 3-epoch references; nine matched 15-epoch runs planned, Powdermill only.")
        return
    if (DURATION_RESULTS / "summary.json").exists():
        raise ValueError("duration results already exist; preserve them")
    models, per_seed, histories, sources = {}, [], [], {}
    for size in SIZES:
        values = {3: [], 15: []}
        for seed in SEEDS:
            previous = plan["baselines"][f"{size}_seed{seed}"]
            before, old_rows, old_score, _ = unpack(Path(previous["report"]))
            path = DURATION_RESULTS / f"seed{seed}" / size / "comparison.json"
            saved, rows, score, protocol = unpack(path)
            actual = torch.load(saved["checkpoint"], map_location="cpu", weights_only=True)
            for key in ("annotations_sha256", "validation_annotations_sha256", "training_recording_ids",
                    "validation_recording_ids", "training_config", "backbone_id", "backbone_revision", "seed",
                    "initial_head_sha256", "target_smoothing", "tv_weight", "hidden", "height", "width", "dropout"):
                if saved[key] != before[key] or actual[key] != saved[key]:
                    raise ValueError(f"unmatched {key}: {size}, seed {seed}")
            for key in ("train_timebins", "validation_timebins", "train_windows", "validation_windows",
                    "checkpoint_selection", "threshold_source"):
                if saved["metrics"][key] != before["metrics"][key]:
                    raise ValueError(f"unmatched {key}: {size}, seed {seed}")
            if (saved["metrics"]["epochs"] != 15 or len(saved["training_history"]) != 15
                    or saved["training_history"] != actual["training_history"]
                    or saved["metrics"] != actual["metrics"]
                    or saved["metrics"]["selected_epoch"] != min(saved["training_history"], key=lambda r: r["validation_loss"])["epoch"]
                    or protocol["partitions"] != plan["partitions"]
                    or digest(saved["checkpoint"]) != saved["checkpoint_sha256"]
                    or protocol["training_duration_pair"]["baseline"] != previous
                    or protocol["training_duration_pair"]["manifest_sha256"] != digest(DURATION_RESULTS / "manifest.json")):
                raise ValueError(f"invalid duration comparison: {size}, seed {seed}")
            for partition, expected in old_rows.items():
                if len(rows[partition]) != len(expected):
                    raise ValueError("different evaluation coverage")
                for row, old in zip(rows[partition], expected):
                    if any(row[k] != old[k] for k in ("name", "group", "seconds")):
                        raise ValueError("different evaluation intervals")
                    if sum(row["area"]["full"]["counts"][0][::2]) != sum(old["area"]["full"]["counts"][0][::2]):
                        raise ValueError("different ground-truth mask")
            if summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score:
                raise ValueError("duration metrics do not reproduce")
            sources[str(path)] = digest(path)
            for epochs, checkpoint, measured in ((3, before, old_score), (15, saved, score)):
                value = dict(seed=seed, **measured, selected_epoch=checkpoint["metrics"]["selected_epoch"],
                    checkpoint=checkpoint["checkpoint"], checkpoint_sha256=checkpoint["checkpoint_sha256"],
                    best_validation_bce=checkpoint["metrics"]["best_val_loss"])
                values[epochs].append(value)
                per_seed.append([size, seed, epochs, value["selected_epoch"], measured["ap"], measured["iou"],
                    measured["threshold"], value["best_validation_bce"]])
            for row in saved["training_history"]:
                histories.append([size, seed, row["epoch"], row["mean_training_loss"], row["validation_loss"]])
        models[size] = {str(epochs): dict(per_seed=records, **{f"{metric}_{stat}": float(function([r[metric] for r in records]))
            for metric in ("ap", "iou") for stat, function in (("mean", np.mean), ("std", lambda x: np.std(x, ddof=1)))})
            for epochs, records in values.items()}
        differences = [new["ap"] - old["ap"] for old, new in zip(values[3], values[15])]
        models[size]["paired_ap_change"] = dict(per_seed=differences, mean=float(np.mean(differences)),
            std=float(np.std(differences, ddof=1)))
    for filename, header, rows in (
            ("per_seed.csv", ["Backbone", "Seed", "Maximum epochs", "Selected epoch", "Mask AP", "2D IoU", "Threshold", "XC validation BCE"], per_seed),
            ("learning_curves.csv", ["Backbone", "Seed", "Epoch", "Training BCE", "XC validation BCE"], histories)):
        with (DURATION_RESULTS / filename).open("w") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
    teacher = json.loads(ANCHOR.read_text())["summary"]["teacher_self_review"]["ap"]
    plot(models, teacher)
    (DURATION_RESULTS / "caption.txt").write_text(
        "Effect of training duration on Powdermill localization at a fixed 10,000 s of self-reviewed XC training labels. "
        "Bars show mean mask AP ±1 sample SD over three training seeds. The best checkpoint within each 3- or 15-epoch "
        "budget is selected by BCE on the same separate 800 s XC validation set. Backbones, optimizer, data and inference "
        "are unchanged. AP averages 35 positive segments of complete Recording_1; per-run IoU thresholds are calibrated "
        "on Recordings_2–4. The dashed line denotes the fixed Qwen teacher. No external evaluation datasets are used.\n")
    write_json(DURATION_RESULTS / "summary.json", dict(models=models, dataset="powdermill", training_seconds=10000,
        validation_seconds=800, evaluation_seconds=10800, seeds=list(SEEDS), sources_sha256=sources,
        teacher_ap=teacher, manifest_sha256=digest(DURATION_RESULTS / "manifest.json")))
    print(json.dumps({size: {budget: {k: v for k, v in row.items() if k != "per_seed"}
        for budget, row in values.items()} for size, values in models.items()}, indent=2))


if __name__ == "__main__":
    main()
