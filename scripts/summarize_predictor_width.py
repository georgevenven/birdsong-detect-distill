#!/usr/bin/env python3
"""Compare d=128 and backbone-matched predictors within the same five-epoch budget."""
import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from birdsong_detect_distill.model import DenseHead
from evaluate_current_backbones import ANCHOR, BACKBONE_WIDTHS, DURATION_RESULTS, REVISIONS, WIDTH_RESULTS, WIDTH_RUN
from summarize_three_seed_figures import unpack


def freeze():
    anchor = json.loads(ANCHOR.read_text())["protocol"]
    protected = {str(ANCHOR): digest(ANCHOR), anchor["split"]: anchor["split_sha256"]}
    split = json.loads(Path(anchor["split"]).read_text())
    for part in split["partitions"].values():
        labels = part["files"]["self_review"]
        protected[labels["path"]] = labels["sha256"]
    for path in ("scripts/train.py", "src/birdsong_detect_distill/model.py", "src/birdsong_detect_distill/data.py",
            "scripts/evaluate_2d.py", "scripts/evaluate_songmae_smoothing.py",
            "src/birdsong_detect_distill/benchmark_data.py", "src/birdsong_detect_distill/benchmark_metrics.py"):
        protected[path] = anchor["code_sha256"][path]
    old_summary = DURATION_RESULTS / "summary.json"
    protected[str(old_summary)] = digest(old_summary)
    sources, configs = {}, {}
    for size, dimension in BACKBONE_WIDTHS.items():
        path = DURATION_RESULTS / "seed0" / size / "comparison.json"
        saved, rows, score, protocol = unpack(path)
        if (saved["seed"] != 0 or saved["metrics"]["epochs"] != 15 or saved["hidden"] != 128
                or protocol["partitions"] != anchor["partitions"]
                or summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score):
            raise ValueError(f"invalid completed reference: {size}")
        best_five = min(saved["training_history"][:5], key=lambda r: r["validation_loss"])["epoch"]
        reusable = best_five == saved["metrics"]["selected_epoch"]
        if reusable != (size in ("micro", "large")):
            raise ValueError("unexpected best-of-five checkpoint reuse plan")
        sources[size] = dict(report=str(path), report_sha256=digest(path), checkpoint=saved["checkpoint"],
            checkpoint_sha256=saved["checkpoint_sha256"], best_epoch_within_five=best_five)
        protected[str(path)] = digest(path)
        protected[saved["checkpoint"]] = saved["checkpoint_sha256"]
        config = AutoConfig.from_pretrained(saved["backbone_id"], revision=REVISIONS[size],
            trust_remote_code=True, local_files_only=True)
        if config.enc_hidden_d != dimension:
            raise ValueError("backbone width differs from its pinned configuration")
        for hidden in sorted({128, dimension}):
            name = f"{size}_d{hidden}"
            reuse = hidden == 128 and reusable
            head = DenseHead(dimension, hidden, 4, 1000, 32, 1, .1)
            configs[name] = dict(size=size, backbone_d=dimension, predictor_d=hidden,
                trainable_parameters=sum(x.numel() for x in head.parameters()), reused=reuse,
                report=str(path) if reuse else str(WIDTH_RESULTS / name / "comparison.json"))
    for path, sha in protected.items():
        if digest(path) != sha:
            raise ValueError(f"protected input or shared implementation changed: {path}")
    plan = dict(dataset="powdermill", training_seconds=10000, validation_seconds=800, seed=0, epochs=5,
        configs=configs, previous=sources, protected_files=protected, partitions=anchor["partitions"],
        code_sha256={path: digest(path) for path in ("scripts/summarize_predictor_width.py",
            "scripts/run_predictor_width.sh", "scripts/evaluate_current_backbones.py")},
        checkpoint_selection="minimum XC validation BCE within epochs 1 through 5",
        reuse="Micro d128 and Large d128 saved checkpoints are also the minima within their first five epochs; "
            "Base d128 is retrained because its best-of-five checkpoint was not saved",
        varying="predictor width only; one transformer layer, four heads, feedforward width 2*d, unchanged input projection design",
        randomness="seed 0 and fixed data membership; changing head dimensions changes RNG consumption and minibatch permutations",
        scope="existing one-self-review labels, no new Qwen calls, no external evaluation, no backbone finetuning")
    path = WIDTH_RESULTS / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen predictor-width experiment changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def plot(models, teacher):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("default")
    plt.rcParams.update({"font.size": 10.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.19, right=.98, bottom=.22, top=.96)
    for condition, offset, color, label in (("control", -.18, ".65", "Predictor d = 128"),
            ("matched", .18, "C0", "Predictor d = backbone")):
        axis.bar(np.arange(3) + offset, [row[condition]["ap"] for row in models.values()],
            width=.33, color=color, label=label)
    axis.axhline(teacher, color=".25", linestyle="--", linewidth=1.2, label="Qwen_teacher labels")
    axis.set(xticks=np.arange(3), xticklabels=[s.title() for s in models], ylim=(0, 1),
        ylabel="Powdermill mask AP", xlabel="SongMAE backbone")
    axis.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.text(.58, .04, "10,000 s · best of 5 epochs · seed 0", ha="center", fontsize=8.5)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(WIDTH_RESULTS / f"predictor_width_ap.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.prepare:
        print("Verified two reusable five-epoch controls; three new training/evaluation runs planned.")
        return
    if (WIDTH_RESULTS / "summary.json").exists():
        raise ValueError("completed predictor-width results exist; preserve them")
    values, sources, table, curves = {}, {}, [], []
    anchor = json.loads(ANCHOR.read_text())["protocol"]
    for name, item in plan["configs"].items():
        path = Path(item["report"])
        saved, rows, score, protocol = unpack(path)
        previous, old_rows, _, _ = unpack(Path(plan["previous"][item["size"]]["report"]))
        actual = torch.load(saved["checkpoint"], map_location="cpu", weights_only=True)
        for key in ("annotations_sha256", "validation_annotations_sha256", "training_recording_ids",
                "validation_recording_ids", "training_config", "backbone_id", "backbone_revision",
                "target_smoothing", "tv_weight", "height", "width", "dropout", "seed"):
            if saved[key] != previous[key] or saved[key] != actual[key]:
                raise ValueError(f"unmatched {key}: {name}")
        if (saved["hidden"] != item["predictor_d"] or actual["hidden"] != saved["hidden"]
                or saved["metrics"]["epochs"] != (15 if item["reused"] else 5)
                or saved["training_history"] != actual["training_history"]
                or saved["metrics"] != actual["metrics"]
                or saved["metrics"]["selected_epoch"] != min(saved["training_history"][:5], key=lambda r: r["validation_loss"])["epoch"]
                or sum(v.numel() for v in actual["head"].values()) != item["trainable_parameters"]
                or digest(saved["checkpoint"]) != saved["checkpoint_sha256"]
                or protocol["partitions"] != plan["partitions"]):
            raise ValueError(f"invalid width/checkpoint/selection: {name}")
        for key in ("train_timebins", "validation_timebins", "train_windows", "validation_windows",
                "checkpoint_selection", "threshold_source"):
            if saved["metrics"][key] != previous["metrics"][key]:
                raise ValueError(f"unmatched {key}: {name}")
        for key in ("coordinate_space", "mask_rule", "threshold_selection", "threshold_grid", "aggregation", "ap_method"):
            if protocol[key] != anchor[key]:
                raise ValueError(f"different scoring procedure: {key}")
        for partition, expected in old_rows.items():
            if len(rows[partition]) != len(expected):
                raise ValueError("incomplete evaluation coverage")
            for row, old in zip(rows[partition], expected):
                if (any(row[k] != old[k] for k in ("name", "group", "seconds"))
                        or sum(row["area"]["full"]["counts"][0][::2]) != sum(old["area"]["full"]["counts"][0][::2])):
                    raise ValueError("different evaluation intervals or ground truth")
        if summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score:
            raise ValueError("metrics do not reproduce")
        value = dict(**score, **item, checkpoint=saved["checkpoint"], checkpoint_sha256=saved["checkpoint_sha256"],
            selected_epoch=saved["metrics"]["selected_epoch"], validation_bce=saved["metrics"]["best_val_loss"])
        values[name] = value
        sources[str(path)] = digest(path)
        table.append([item["size"], item["predictor_d"], item["trainable_parameters"], value["selected_epoch"],
            score["ap"], score["iou"], score["threshold"], item["reused"]])
        for row in saved["training_history"][:5]:
            curves.append([name, row["epoch"], row["mean_training_loss"], row["validation_loss"]])
    models = {size: dict(control=values[f"{size}_d128"], matched=values[f"{size}_d{dimension}"])
        for size, dimension in BACKBONE_WIDTHS.items()}
    teacher = json.loads(ANCHOR.read_text())["summary"]["teacher_self_review"]["ap"]
    for filename, header, data in (("comparison.csv", ["Backbone", "Predictor d", "Trainable parameters", "Selected epoch", "Mask AP", "2D IoU", "Threshold", "Reused checkpoint"], table),
            ("learning_curves.csv", ["Configuration", "Epoch", "Training BCE", "XC validation BCE"], curves)):
        with (WIDTH_RESULTS / filename).open("w") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)
    plot(models, teacher)
    (WIDTH_RESULTS / "caption.txt").write_text(
        "Predictor-width ablation with fixed SongMAE backbones, 10,000 s of self-reviewed XC training labels, "
        "and seed 0. Compare d=128 with d matched to each backbone (Micro 128, Base 384, Large 768). "
        "Both conditions select the minimum-BCE checkpoint within five epochs using the same recording-disjoint "
        "800 s XC validation set. Micro is the same condition in both bars. AP averages the 35 positive segments "
        "of complete Powdermill Recording_1. Inference and per-model threshold calibration on Recordings_2–4 "
        "are unchanged. Single-seed results; no statistical significance is implied.\n")
    write_json(WIDTH_RESULTS / "summary.json", dict(models=models, configurations=values, dataset="powdermill",
        training_seconds=10000, validation_seconds=800, epochs=5, seed=0, teacher_ap=teacher,
        sources_sha256=sources, manifest_sha256=digest(WIDTH_RESULTS / "manifest.json")))
    print(json.dumps({size: {condition: {k: row[k] for k in ("predictor_d", "ap", "iou", "selected_epoch")}
        for condition, row in pair.items()} for size, pair in models.items()}, indent=2))


if __name__ == "__main__":
    main()
