#!/usr/bin/env python3
"""Isolate predictor width with a frozen Large backbone and best-of-five XC selection."""
import argparse
import csv
import json
from pathlib import Path

import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from evaluate_current_backbones import ANCHOR, LARGE_WIDTH_RESULTS as OUT, WIDTH_RESULTS
from summarize_three_seed_figures import unpack


def freeze():
    previous_path = WIDTH_RESULTS / "summary.json"
    previous = json.loads(previous_path.read_text())
    anchor = json.loads(ANCHOR.read_text())["protocol"]
    protected = {str(ANCHOR): digest(ANCHOR), str(previous_path): digest(previous_path),
        anchor["split"]: anchor["split_sha256"]}
    for part in json.loads(Path(anchor["split"]).read_text())["partitions"].values():
        labels = part["files"]["self_review"]
        protected[labels["path"]] = labels["sha256"]
    protected.update({p: sha for p, sha in anchor["code_sha256"].items()
        if p in ("scripts/train.py", "scripts/evaluate_2d.py", "scripts/evaluate_songmae_smoothing.py")
        or p.startswith("src/")})
    reports = {"384": str(OUT / "large_d384/comparison.json")}
    for width in (128, 768):
        value = previous["configurations"][f"large_d{width}"]
        path = Path(value["report"])
        saved, _, _, _ = unpack(path)
        if saved["metrics"]["selected_epoch"] != min(saved["training_history"][:5],
                key=lambda r: r["validation_loss"])["epoch"]:
            raise ValueError("existing checkpoint is not the best within five epochs")
        reports[str(width)] = str(path)
        protected[str(path)] = digest(path)
        protected[saved["checkpoint"]] = saved["checkpoint_sha256"]
    for suffix in ("png", "pdf", "svg"):
        path = WIDTH_RESULTS / f"predictor_width_ap.{suffix}"
        protected[str(path)] = digest(path)
    for path, sha in protected.items():
        if digest(path) != sha:
            raise ValueError(f"protected input changed: {path}")
    plan = dict(backbone="georgeven/songmae-large-32x1", backbone_d=768,
        training_seconds=10000, validation_seconds=800, seed=0, epochs=5, reports=reports,
        protected_files=protected, partitions=anchor["partitions"],
        checkpoint_selection="minimum XC validation BCE within epochs 1 through 5",
        reuse="Reuse matching d128 and d768 checkpoints; train only d384",
        randomness="Fixed seed/data membership; head width changes RNG consumption and minibatch permutations",
        scope="Powdermill only; same one-self-review labels, frozen backbone, hard BCE, probability smoothing",
        code_sha256={p: digest(p) for p in ("scripts/summarize_large_predictor_width.py",
            "scripts/run_large_predictor_width.sh", "scripts/evaluate_current_backbones.py")})
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen fixed-Large experiment changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.prepare:
        print("Verified matching d128/d768 references; one new Large+d384 run planned.")
        return
    if (OUT / "summary.json").exists():
        raise ValueError("completed comparison exists; preserve it")
    baseline, old_rows, _, old_protocol = unpack(Path(plan["reports"]["128"]))
    models, sources = {}, {}
    for width in (128, 384, 768):
        path = Path(plan["reports"][str(width)])
        saved, rows, score, protocol = unpack(path)
        actual = torch.load(saved["checkpoint"], map_location="cpu", weights_only=True)
        if (digest(saved["checkpoint"]) != saved["checkpoint_sha256"]
                or saved["hidden"] != width or actual["hidden"] != width
                or saved["metrics"] != actual["metrics"]
                or saved["training_history"] != actual["training_history"]
                or saved["metrics"]["epochs"] != (15 if width == 128 else 5)
                or saved["metrics"]["selected_epoch"] != min(saved["training_history"][:5],
                    key=lambda r: r["validation_loss"])["epoch"]):
            raise ValueError(f"invalid checkpoint/width/selection: {width}")
        for key in ("annotations_sha256", "validation_annotations_sha256", "training_recording_ids",
                "validation_recording_ids", "backbone_id", "backbone_revision", "training_config",
                "target_smoothing", "tv_weight", "height", "width", "dropout", "seed"):
            if saved[key] != baseline[key] or saved[key] != actual[key]:
                raise ValueError(f"unmatched training: {key}")
        for key in ("train_timebins", "validation_timebins", "train_windows", "validation_windows",
                "checkpoint_selection", "threshold_source"):
            if saved["metrics"][key] != baseline["metrics"][key]:
                raise ValueError(f"unmatched training budget: {key}")
        for key in ("partitions", "student_inference", "coordinate_space", "mask_rule", "threshold_selection",
                "threshold_grid", "aggregation", "ap_method"):
            if protocol[key] != old_protocol[key]:
                raise ValueError(f"unmatched evaluation: {key}")
        for source, sha in old_protocol["code_sha256"].items():
            if source != "scripts/evaluate_current_backbones.py" and protocol["code_sha256"].get(source) != sha:
                raise ValueError(f"shared implementation differs: {source}")
        for partition, expected in old_rows.items():
            if len(rows[partition]) != len(expected):
                raise ValueError("incomplete evaluation")
            for row, old in zip(rows[partition], expected):
                if (any(row[k] != old[k] for k in ("name", "group", "seconds"))
                        or sum(row["area"]["full"]["counts"][0][::2]) != sum(old["area"]["full"]["counts"][0][::2])):
                    raise ValueError("different ground truth/coverage")
        if summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score:
            raise ValueError("calibration/metrics do not reproduce")
        models[str(width)] = dict(**score, selected_epoch=saved["metrics"]["selected_epoch"],
            trainable_parameters=sum(v.numel() for v in actual["head"].values()),
            validation_bce=saved["metrics"]["best_val_loss"], checkpoint=saved["checkpoint"],
            checkpoint_sha256=saved["checkpoint_sha256"])
        sources[str(path)] = digest(path)
    with (OUT / "comparison.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["Predictor d", "Trainable parameters", "Selected epoch", "Mask AP", "2D IoU", "Threshold"])
        writer.writerows([d, v["trainable_parameters"], v["selected_epoch"], v["ap"], v["iou"], v["threshold"]]
            for d, v in models.items())
    plot(models)
    write_json(OUT / "summary.json", dict(models=models, protocol=plan,
        sources_sha256=sources, manifest_sha256=digest(OUT / "manifest.json")))
    print(json.dumps(models, indent=2))


def plot(models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.19, right=.97, bottom=.22, top=.89)
    bars = ax.bar(list(models), [v["ap"] for v in models.values()], color=[".65", "C0", "C0"], width=.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set(ylim=(0, 1), ylabel="Powdermill mask AP", xlabel="Predictor width (d)", title="Frozen SongMAE-Large")
    fig.text(.58, .04, "10,000 s · best of 5 epochs · seed 0", ha="center", fontsize=8.5)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"large_predictor_width_ap.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
