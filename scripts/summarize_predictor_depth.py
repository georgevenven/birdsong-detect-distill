#!/usr/bin/env python3
"""Compare a two-layer d384 predictor with the completed one-layer Large controls."""
import argparse
import csv
import json
from pathlib import Path

import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from birdsong_detect_distill.model import DenseHead
from evaluate_current_backbones import DEPTH_RESULTS as OUT, LARGE_WIDTH_RESULTS
from summarize_three_seed_figures import unpack


CHANGED = ("src/birdsong_detect_distill/model.py", "scripts/train.py", "scripts/evaluate_2d.py")


def freeze():
    previous_path = LARGE_WIDTH_RESULTS / "summary.json"
    previous = json.loads(previous_path.read_text())
    protected = {p: sha for p, sha in previous["protocol"]["protected_files"].items() if p not in CHANGED}
    protected.update(previous["sources_sha256"])
    protected[str(previous_path)] = digest(previous_path)
    changes = {}
    for path in CHANGED:
        source = OUT / "source_before" / path
        sha = previous["protocol"]["protected_files"][path]
        protected[str(source)] = sha
        changes[path] = dict(before=str(source), before_sha256=sha, after_sha256=digest(path))
    for suffix in ("png", "pdf", "svg"):
        path = LARGE_WIDTH_RESULTS / f"large_predictor_width_ap.{suffix}"
        protected[str(path)] = digest(path)
    configs = {}
    for name, layers, width in (("1x128", 1, 128), ("1x384", 1, 384), ("2x384", 2, 384), ("1x768", 1, 768)):
        report = str(OUT / "large_d384_l2/comparison.json") if layers == 2 else previous["protocol"]["reports"][str(width)]
        configs[name] = dict(layers=layers, width=width, report=report, saved_epochs=15 if width == 128 else 5)
        if layers == 1:
            saved, rows, score, _ = unpack(Path(report))
            if (saved.get("head_layers", 1) != 1 or saved["hidden"] != width
                    or saved["metrics"]["selected_epoch"] != min(saved["training_history"][:5],
                        key=lambda r: r["validation_loss"])["epoch"]
                    or summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score):
                raise ValueError("invalid completed one-layer reference")
            protected[saved["checkpoint"]] = saved["checkpoint_sha256"]
    for path, sha in protected.items():
        if digest(path) != sha:
            raise ValueError(f"protected input changed: {path}")
    _, _, _, reference = unpack(Path(configs["1x384"]["report"]))
    code = [*reference["code_sha256"], "scripts/summarize_predictor_depth.py", "scripts/run_predictor_depth.sh"]
    plan = dict(backbone="georgeven/songmae-large-32x1", backbone_d=768, training_seconds=10000,
        validation_seconds=800, seed=0, epochs=5, configurations=configs, protected_files=protected,
        partitions=reference["partitions"], code_sha256={p: digest(p) for p in code}, architecture_changes=changes,
        checkpoint_selection="minimum XC validation BCE within epochs 1 through 5",
        reuse="Completed one-layer controls; d128 saved epoch 3 is also its minimum within five epochs",
        initialization="Independent second block, not a clone; one-layer keys and initialization remain unchanged",
        randomness="Seed and data membership fixed; additional parameters change RNG consumption and minibatch order",
        scope="Powdermill only, frozen Large, one-self-review labels, hard BCE, unchanged probability smoothing and calibration")
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen depth experiment changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.prepare:
        print("Verified three matching one-layer controls; train only Large + two d384 layers.")
        return
    if (OUT / "summary.json").exists():
        raise ValueError("completed comparison exists; preserve it")
    baseline, expected, _, reference = unpack(Path(plan["configurations"]["1x384"]["report"]))
    models, sources = {}, {}
    for name, config in plan["configurations"].items():
        path = Path(config["report"])
        saved, rows, score, protocol = unpack(path)
        actual = torch.load(saved["checkpoint"], map_location="cpu", weights_only=True)
        if (digest(saved["checkpoint"]) != saved["checkpoint_sha256"]
                or saved["hidden"] != config["width"] or actual["hidden"] != config["width"]
                or saved.get("head_layers", 1) != config["layers"] or actual.get("head_layers", 1) != config["layers"]
                or saved["metrics"] != actual["metrics"] or saved["training_history"] != actual["training_history"]
                or saved["metrics"]["epochs"] != config["saved_epochs"]
                or saved["metrics"]["selected_epoch"] != min(saved["training_history"][:5],
                    key=lambda r: r["validation_loss"])["epoch"]):
            raise ValueError(f"invalid checkpoint/architecture/selection: {name}")
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
            if protocol[key] != reference[key]:
                raise ValueError(f"unmatched evaluation: {key}")
        implementation = plan["code_sha256"] if config["layers"] == 2 else reference["code_sha256"]
        for source in reference["code_sha256"]:
            if source != "scripts/evaluate_current_backbones.py" and protocol["code_sha256"][source] != implementation[source]:
                raise ValueError(f"unmatched implementation: {source}")
        for partition, old_rows in expected.items():
            if len(rows[partition]) != len(old_rows):
                raise ValueError("incomplete coverage")
            for row, old in zip(rows[partition], old_rows):
                if (any(row[k] != old[k] for k in ("name", "group", "seconds"))
                        or sum(row["area"]["full"]["counts"][0][::2]) != sum(old["area"]["full"]["counts"][0][::2])):
                    raise ValueError("different ground truth/coverage")
        if summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score:
            raise ValueError("metrics/calibration do not reproduce")
        count = sum(v.numel() for v in actual["head"].values())
        head = DenseHead(768, config["width"], 4, 1000, 32, 1, .1, config["layers"])
        head.load_state_dict(actual["head"])
        if count != sum(p.numel() for p in head.parameters()):
            raise ValueError("incorrect parameter count")
        models[name] = dict(**score, layers=config["layers"], width=config["width"], trainable_parameters=count,
            selected_epoch=saved["metrics"]["selected_epoch"], validation_bce=saved["metrics"]["best_val_loss"],
            checkpoint=saved["checkpoint"], checkpoint_sha256=saved["checkpoint_sha256"])
        sources[str(path)] = digest(path)
    with (OUT / "comparison.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["Layers", "Predictor d", "Parameters", "Selected epoch", "Pixel AP", "2D IoU", "Threshold"])
        writer.writerows([v[k] for k in ("layers", "width", "trainable_parameters", "selected_epoch", "ap", "iou", "threshold")]
            for v in models.values())
    plot(models)
    write_json(OUT / "summary.json", dict(models=models, protocol=plan, sources_sha256=sources,
        manifest_sha256=digest(OUT / "manifest.json")))
    print(json.dumps(models, indent=2))


def plot(models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.19, right=.97, bottom=.22, top=.89)
    bars = ax.bar([name.replace("x", "×") for name in models], [v["ap"] for v in models.values()],
        color=["C0" if v["layers"] == 2 else ".65" for v in models.values()], width=.65)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(ylim=(0, 1), ylabel="Powdermill pixel AP", xlabel="Predictor layers × width", title="Frozen SongMAE-Large")
    fig.text(.58, .04, "10,000 s · best of 5 epochs · seed 0", ha="center", fontsize=8.5)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"predictor_depth_ap.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
