#!/usr/bin/env python3
"""Freeze and summarize paired ablations against the hard-mask Large+d384 baseline."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from birdsong_detect_distill.data import PixelWindows, read_rows
from evaluate_current_backbones import CONFIDENCE_RESULTS, DEPTH_RESULTS, LARGE_WIDTH_RESULTS, LR_RESULTS, UNCERTAIN_RESULTS
from summarize_three_seed_figures import unpack


BASELINE = LARGE_WIDTH_RESULTS / "large_d384/comparison.json"
OUT = UNCERTAIN_RESULTS
CHANGED = ("scripts/train.py", "src/birdsong_detect_distill/model.py", "src/birdsong_detect_distill/data.py")


def mask_audit(baseline, split, confidence_targets=False):
    config = AutoConfig.from_pretrained(baseline["backbone_id"], revision=baseline["backbone_revision"],
        trust_remote_code=True, local_files_only=True)
    counts = {}
    for part, seconds in (("train", 10000), ("validation", 800)):
        rows = read_rows(split["partitions"][part]["files"]["self_review"]["path"], seed=0)
        data = PixelWindows(rows, "data/xcl/shards", config, ignore_uncertain=not confidence_targets,
            confidence_targets=confidence_targets)
        if sum(w[2] for w in data.windows) != seconds * 200:
            raise ValueError("audio budget changed")
        overlaps, foreground_pixels, target_sum, maximum = 0, 0, 0.0, 0.0
        for row, start, valid, target in data.windows:
            certain, uncertain = np.zeros((128, valid), bool), np.zeros((128, valid), bool)
            confidence_map = np.zeros((128, valid), np.float32) if confidence_targets else None
            for box in row["boxes"]:
                left, right = max(start, box["start_timebin"]), min(start + valid, box["end_timebin"])
                if left < right:
                    mask = uncertain if box["label"] == "uncertain_vocalization" else certain
                    mask[box["low_mel_bin"]:box["high_mel_bin"], left - start:right - start] = True
                    if confidence_targets:
                        score = float(box["confidence"])
                        if not np.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("invalid confidence")
                        region = confidence_map[box["low_mel_bin"]:box["high_mel_bin"], left - start:right - start]
                        region[:] = np.where(region > score, region, np.float32(score))
            expected = confidence_map if confidence_targets else np.where(certain, 1, np.where(uncertain, -1, 0))
            if not np.array_equal(target[:, :valid], expected) or target[:, valid:].any():
                raise ValueError("uncertainty/overlap/padding mask is incorrect")
            overlaps += int((certain & uncertain).sum())
            foreground_pixels += int((certain | uncertain).sum())
            target_sum += float(target[:, :valid].sum(dtype=np.float64))
            maximum = max(maximum, float(target.max()))
        stats = data.supervision_counts() if not confidence_targets else dict(valid_pixels=seconds * 200 * 128,
            foreground_pixels=foreground_pixels, background_pixels=seconds * 200 * 128 - foreground_pixels,
            ignored_pixels=0, target_sum=target_sum, foreground_mean_target=target_sum / foreground_pixels, maximum_target=maximum)
        counts[part] = dict(**stats, retained_positive_overlap_pixels=overlaps,
            windows=len(data), recording_ids=sorted({r["recording"] for r in rows}), timebins=seconds * 200)
        if not confidence_targets and (counts[part]["fully_ignored_windows"] or not counts[part]["ignored_pixels"]):
            raise ValueError("unexpected absent uncertainty or fully ignored windows")
    return counts


def freeze(confidence_targets=False):
    predecessor = UNCERTAIN_RESULTS if confidence_targets else DEPTH_RESULTS
    old = json.loads((predecessor / "manifest.json").read_text())
    changed = (*CHANGED, "scripts/summarize_uncertain_ablation.py") if confidence_targets else CHANGED
    baseline, _, _, reference = unpack(BASELINE)
    protected = {p: sha for p, sha in old["protected_files"].items() if p not in changed}
    protected[str(predecessor / "manifest.json")] = digest(predecessor / "manifest.json")
    protected[str(BASELINE)] = digest(BASELINE)
    protected[baseline["checkpoint"]] = baseline["checkpoint_sha256"]
    for path in changed:
        protected[str(OUT / "source_before" / path)] = old["code_sha256"][path]
    for source, sha in old["code_sha256"].items():
        if source not in changed and source != "scripts/evaluate_current_backbones.py":
            protected[source] = sha
    for folder, stem in ((DEPTH_RESULTS, "predictor_depth_ap"), (LARGE_WIDTH_RESULTS, "large_predictor_width_ap")):
        for name in ("summary.json", "comparison.csv", *(f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg"))):
            protected[str(folder / name)] = digest(folder / name)
    if confidence_targets:
        previous_summary = json.loads((UNCERTAIN_RESULTS / "summary.json").read_text())
        protected.update(previous_summary["sources_sha256"])
        for value in previous_summary["models"].values():
            protected[value["checkpoint"]] = value["checkpoint_sha256"]
        for name in ("summary.json", "comparison.csv", *(f"uncertain_supervision_ap.{s}" for s in ("png", "pdf", "svg"))):
            protected[str(UNCERTAIN_RESULTS / name)] = digest(UNCERTAIN_RESULTS / name)
    for path, sha in protected.items():
        if digest(path) != sha:
            raise ValueError(f"protected input/code changed: {path}")
    split_path = json.loads(Path(reference["anchor"]).read_text())["protocol"]["split"]
    counts = mask_audit(baseline, json.loads(Path(split_path).read_text()), confidence_targets)
    runner = "scripts/run_confidence_ablation.sh" if confidence_targets else "scripts/run_uncertain_ablation.sh"
    code = [*reference["code_sha256"], "scripts/summarize_uncertain_ablation.py", runner]
    name = "large_d384_confidence" if confidence_targets else "large_d384_ignore_uncertain"
    plan = dict(baseline=str(BASELINE), new_report=str(OUT / name / "comparison.json"),
        backbone=baseline["backbone_id"], width=384, layers=1, training_seconds=10000, validation_seconds=800,
        epochs=5, seed=0, supervision=counts, partitions=reference["partitions"],
        protected_files=protected, code_sha256={p: digest(p) for p in code},
        label_policy="Uncertain-only pixels ignored in both XC training and validation; target/chorus overlaps stay positive",
        checkpoint_selection="minimum XC validation BCE over supervised pixels within five epochs",
        validation_caution="BCE values across policies use different pixel sets and are not directly comparable",
        controlled="Same architecture, initial weights, all audio windows, minibatch order, optimizer, and Powdermill scoring",
        scope="Powdermill only; Qwen remains paused; no new labels or external evaluation")
    if confidence_targets:
        plan.update(label_policy="Maximum Qwen confidence inside target/uncertain/chorus boxes; zero outside; BCE targets, not loss weights",
            checkpoint_selection="minimum confidence-target XC validation BCE within five epochs",
            validation_caution="Hard and confidence targets define different objectives; validation BCE values are not directly comparable",
            confidence_caution="Qwen self-reported label confidences are not empirically calibrated bird probabilities")
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen uncertain-label experiment changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def freeze_learning_rate():
    previous = json.loads((CONFIDENCE_RESULTS / "manifest.json").read_text())
    baseline, _, _, reference = unpack(BASELINE)
    changed = {"scripts/train.py", "scripts/evaluate_current_backbones.py", "scripts/summarize_uncertain_ablation.py"}
    protected = {p: sha for p, sha in previous["protected_files"].items() if p not in changed}
    protected.update({p: sha for p, sha in previous["code_sha256"].items() if p not in changed})
    protected.update({str(OUT / "source_before" / p):previous["code_sha256"][p] for p in changed})
    protected[str(CONFIDENCE_RESULTS / "manifest.json")] = digest(CONFIDENCE_RESULTS / "manifest.json")
    protected[str(BASELINE)] = digest(BASELINE)
    protected[baseline["checkpoint"]] = baseline["checkpoint_sha256"]
    last = json.loads((CONFIDENCE_RESULTS / "summary.json").read_text())
    protected.update(last["sources_sha256"])
    for value in last["models"].values():
        protected[value["checkpoint"]] = value["checkpoint_sha256"]
    for path in CONFIDENCE_RESULTS.iterdir():
        if path.is_file():
            protected[str(path)] = digest(path)
    for path, sha in protected.items():
        if digest(path) != sha:
            raise ValueError(f"protected input changed: {path}")
    code = [*reference["code_sha256"], "scripts/summarize_uncertain_ablation.py", "scripts/run_learning_rate_ablation.sh"]
    plan = dict(baseline=str(BASELINE), new_report=str(OUT / "large_d384_lr3e-4/comparison.json"),
        backbone=baseline["backbone_id"], width=384, layers=1, training_seconds=10000, validation_seconds=800,
        epochs=5, seed=0, learning_rates=[1e-3, 3e-4], partitions=reference["partitions"],
        protected_files=protected, code_sha256={p:digest(p) for p in code},
        label_policy="Hard foreground union: target, uncertain and chorus; no softening, TV or confidence weighting",
        checkpoint_selection="minimum XC validation BCE within five epochs",
        controlled="Only AdamW learning rate changes; same initial head, minibatch order, data and evaluation",
        caveat="Single seed; equal epochs, not a convergence comparison; Powdermill is development data",
        scope="Powdermill only; start after YOLO11l finishes; never restart Qwen")
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen learning-rate experiment changed")
    if not path.exists():
        write_json(path, plan)
    return plan


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--confidence-targets", action="store_true")
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,))
    args = parser.parse_args()
    if args.learning_rate and args.confidence_targets:
        parser.error("do not combine target-policy and learning-rate ablations")
    OUT = LR_RESULTS if args.learning_rate else CONFIDENCE_RESULTS if args.confidence_targets else UNCERTAIN_RESULTS
    plan = freeze_learning_rate() if args.learning_rate else freeze(args.confidence_targets)
    if args.prepare:
        print(json.dumps({k:v for k,v in plan.items() if k not in ("protected_files", "code_sha256", "supervision")}, indent=2)
            if args.learning_rate else json.dumps({part: {k: v for k, v in counts.items() if k != "recording_ids"}
                for part, counts in plan["supervision"].items()}, indent=2))
        return
    if (OUT / "summary.json").exists():
        raise ValueError("completed results exist; preserve them")
    baseline, expected, _, reference = unpack(BASELINE)
    models, sources = {}, {}
    baseline_name = "1e-3" if args.learning_rate else "Included"
    new_name = "3e-4" if args.learning_rate else "Confidence" if args.confidence_targets else "Ignored"
    for name, path in ((baseline_name, BASELINE), (new_name, Path(plan["new_report"]))):
        saved, rows, score, protocol = unpack(path)
        actual = torch.load(saved["checkpoint"], map_location="cpu", weights_only=True)
        ignore = name == "Ignored"
        confidence = name == "Confidence"
        if (digest(saved["checkpoint"]) != saved["checkpoint_sha256"]
                or saved.get("ignore_uncertain", False) != ignore or actual.get("ignore_uncertain", False) != ignore
                or saved.get("confidence_targets", False) != confidence or actual.get("confidence_targets", False) != confidence
                or saved.get("head_layers", 1) != 1 or actual.get("head_layers", 1) != 1
                or saved["metrics"] != actual["metrics"] or saved["training_history"] != actual["training_history"]
                or saved["metrics"]["epochs"] != 5 or len(saved["training_history"]) != 5
                or saved["metrics"]["selected_epoch"] != min(saved["training_history"], key=lambda r: r["validation_loss"])["epoch"]):
            raise ValueError("invalid checkpoint/policy/selection")
        for key in ("annotations_sha256", "validation_annotations_sha256", "training_recording_ids", "validation_recording_ids",
                "backbone_id", "backbone_revision", "initial_head_sha256", "hidden", "height", "width",
                "dropout", "seed", "target_smoothing", "tv_weight"):
            if saved[key] != baseline[key] or saved[key] != actual[key]:
                raise ValueError(f"unmatched setup: {key}")
        config = dict(baseline["training_config"])
        if args.learning_rate and name == new_name:
            config["learning_rate"] = args.learning_rate
        if saved["training_config"] != config or actual["training_config"] != config:
            raise ValueError("unmatched optimizer configuration")
        for new, old in zip(saved["training_history"], baseline["training_history"]):
            for key in ("epoch", "sample_order_sha256", "cpu_rng_sha256"):
                if new[key] != old[key]:
                    raise ValueError(f"unmatched training randomness: {key}")
        for key in ("train_timebins", "validation_timebins", "train_windows", "validation_windows", "checkpoint_selection", "threshold_source"):
            if saved["metrics"][key] != baseline["metrics"][key]:
                raise ValueError(f"unmatched budget: {key}")
        if ignore:
            if saved["supervision"] != actual["supervision"]:
                raise ValueError("checkpoint supervision metadata differs")
            for part, counts in saved["supervision"].items():
                if any(plan["supervision"][part][key] != value for key, value in counts.items()):
                    raise ValueError("supervision counts differ from audited masks")
        for key in ("partitions", "student_inference", "coordinate_space", "mask_rule", "threshold_selection", "threshold_grid", "aggregation", "ap_method"):
            if protocol[key] != reference[key]:
                raise ValueError(f"unmatched evaluation: {key}")
        code = plan["code_sha256"] if name != baseline_name else reference["code_sha256"]
        if any(code[p] != sha for p, sha in protocol["code_sha256"].items()):
            raise ValueError("unmatched implementation")
        for part, old_rows in expected.items():
            if len(rows[part]) != len(old_rows):
                raise ValueError("incomplete coverage")
            for row, old in zip(rows[part], old_rows):
                if (any(row[k] != old[k] for k in ("name", "group", "seconds"))
                        or sum(row["area"]["full"]["counts"][0][::2]) != sum(old["area"]["full"]["counts"][0][::2])):
                    raise ValueError("different human labels/coverage")
        if summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"] != score:
            raise ValueError("metrics/calibration do not reproduce")
        models[name] = dict(**score, selected_epoch=saved["metrics"]["selected_epoch"], checkpoint=saved["checkpoint"],
            checkpoint_sha256=saved["checkpoint_sha256"], validation_bce=saved["metrics"]["best_val_loss"])
        sources[str(path)] = digest(path)
    with (OUT / "comparison.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["Learning rate" if args.learning_rate else "Target policy", "Pixel AP", "2D IoU", "Precision", "Recall", "Threshold", "Epoch"])
        writer.writerows([name, *(v[k] for k in ("ap", "iou", "pooled_precision", "pooled_recall", "threshold", "selected_epoch"))]
            for name, v in models.items())
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
    confidence = "Confidence" in models
    learning_rate = "3e-4" in models
    labels = ["Hard masks", "Qwen scores"] if confidence else list(models)
    bars = ax.bar(labels, [v["ap"] for v in models.values()], color=[".65", "C0"], width=.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set(ylim=(0, 1), ylabel="Powdermill pixel AP",
        xlabel="Learning rate" if learning_rate else "BCE target" if confidence else "Uncertain-only pixels", title="Frozen Large · 1 × 384")
    fig.text(.58, .04, "10,000 s · best of 5 epochs · seed 0", ha="center", fontsize=8.5)
    for suffix in ("png", "pdf", "svg"):
        stem = "learning_rate_ap" if learning_rate else "confidence_targets_ap" if confidence else "uncertain_supervision_ap"
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
