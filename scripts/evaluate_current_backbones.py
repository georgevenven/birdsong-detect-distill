#!/usr/bin/env python3
"""Match backbone sizes or nested label budgets to the completed Large experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, THRESHOLDS, area, calibrate, summarize
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import load_heads, songmae_probabilities
from evaluate_songmae_smoothing import smooth, truth_and_coverage


RUN = "backbones_10000train_800val_2026-09-10"
ARTIFACTS = Path("artifacts") / RUN
RESULTS = Path("results/qwen_teacher_powdermill") / RUN
ANCHOR = Path("results/qwen_teacher_powdermill/stage_pipeline_10000train_800val_2026-09-09/comparison.json")
REVISIONS = {"micro": "78a7f7e959e76d791d6ee6d8f761cb139c8654e4",
    "base": "d47970abb03ad01c575bd98ed21932283d00eaa0",
    "large": "f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e"}
SCALING_RUN = "scaling_10000train_800val_2026-09-10"
SCALING_RESULTS = Path("results/qwen_teacher_powdermill") / SCALING_RUN
SCALING_LABELS = Path("data/annotations/xcl") / SCALING_RUN
MULTISEED_RUN = "scaling_backbones_three_seeds_2026-09-10"
MULTISEED_RESULTS = Path("results/qwen_teacher_powdermill") / MULTISEED_RUN
DURATION_RUN = "training_duration_10000train_800val_2026-09-10"
DURATION_RESULTS = Path("results/qwen_teacher_powdermill") / DURATION_RUN
WIDTH_RUN = "predictor_width_10000train_800val_2026-09-10"
WIDTH_RESULTS = Path("results/qwen_teacher_powdermill") / WIDTH_RUN
LARGE_WIDTH_RUN = "large_predictor_width_10000train_800val_2026-09-10"
LARGE_WIDTH_RESULTS = Path("results/qwen_teacher_powdermill") / LARGE_WIDTH_RUN
DEPTH_RUN = "predictor_depth_10000train_800val_2026-09-10"
DEPTH_RESULTS = Path("results/qwen_teacher_powdermill") / DEPTH_RUN
UNCERTAIN_RUN = "uncertain_ignore_10000train_800val_2026-09-10"
UNCERTAIN_RESULTS = Path("results/qwen_teacher_powdermill") / UNCERTAIN_RUN
CONFIDENCE_RUN = "confidence_targets_10000train_800val_2026-09-10"
CONFIDENCE_RESULTS = Path("results/qwen_teacher_powdermill") / CONFIDENCE_RUN
LR_RUN = "learning_rate_10000train_800val_2026-09-10"
LR_RESULTS = Path("results/qwen_teacher_powdermill") / LR_RUN
BACKBONE_WIDTHS = {"micro": 128, "base": 384, "large": 768}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=REVISIONS, required=True)
    parser.add_argument("--seconds", type=int, choices=(100, 1000, 10000), default=10000)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--training-epochs", type=int, choices=(3, 5, 15), default=3)
    parser.add_argument("--head-width", type=int, choices=(128, 384, 768), default=128)
    parser.add_argument("--head-layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fixed-large-width", action="store_true")
    parser.add_argument("--ignore-uncertain", action="store_true")
    parser.add_argument("--confidence-targets", action="store_true")
    parser.add_argument("--learning-rate", type=float, choices=(1e-3, 3e-4), default=1e-3)
    parser.add_argument("--prediction-cache", choices=("all", "samples"), default="all")
    parser.add_argument("--root", type=Path, default=Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw"))
    args = parser.parse_args()
    lr_ablation = args.learning_rate != 1e-3
    if lr_ablation and (args.confidence_targets or args.ignore_uncertain or args.size != "large"
            or args.head_width != 384 or args.head_layers != 1 or not args.fixed_large_width):
        parser.error("learning-rate ablation requires hard BCE, fixed Large and one d384 layer")
    if args.confidence_targets and (args.ignore_uncertain or args.size != "large" or args.head_width != 384 or args.head_layers != 1 or not args.fixed_large_width):
        parser.error("confidence-target ablation requires fixed Large with one d384 layer and no ignored uncertainty")
    if args.ignore_uncertain and (args.size != "large" or args.head_width != 384 or args.head_layers != 1 or not args.fixed_large_width):
        parser.error("uncertain-label ablation requires fixed Large with one d384 layer")
    if args.head_layers == 2 and (args.size != "large" or args.head_width != 384 or not args.fixed_large_width):
        parser.error("depth ablation requires --size large --head-width 384 --fixed-large-width")
    if args.fixed_large_width and (args.size != "large" or args.training_epochs != 5):
        parser.error("the fixed-Large width study requires Large and five epochs")
    width_run = LARGE_WIDTH_RUN if args.fixed_large_width else WIDTH_RUN
    width_results = LARGE_WIDTH_RESULTS if args.fixed_large_width else WIDTH_RESULTS
    if args.head_layers == 2:
        width_run, width_results = DEPTH_RUN, DEPTH_RESULTS
    if args.ignore_uncertain:
        width_run, width_results = UNCERTAIN_RUN, UNCERTAIN_RESULTS
    if args.confidence_targets:
        width_run, width_results = CONFIDENCE_RUN, CONFIDENCE_RESULTS
    if lr_ablation:
        width_run, width_results = LR_RUN, LR_RESULTS
    if args.head_width != 128 and args.training_epochs != 5:
        parser.error("wider predictors belong to the five-epoch width experiment")
    out, cache = RESULTS / args.size, ARTIFACTS / "predictions" / args.size
    checkpoint = ARTIFACTS / f"{args.size}.pt"
    budget = None
    if args.seconds != 10000:
        if args.size != "large":
            parser.error("label-budget comparison uses Large only")
        budget = json.loads((SCALING_LABELS / "manifest.json").read_text())["budgets"][str(args.seconds)]
        checkpoint = Path("artifacts") / SCALING_RUN / f"large_s{args.seconds}.pt"
        out = SCALING_RESULTS / f"s{args.seconds}"
        cache = checkpoint.parent / "predictions" / f"s{args.seconds}"
    if args.seed:
        name = f"{args.size}_s{args.seconds}"
        out = MULTISEED_RESULTS / f"seed{args.seed}" / name
        checkpoint = Path("artifacts") / MULTISEED_RUN / f"seed{args.seed}" / f"{name}.pt"
        cache = checkpoint.parent / "predictions" / name
    if args.training_epochs == 15:
        if args.seconds != 10000:
            parser.error("the duration experiment uses exactly 10,000 training seconds")
        out = DURATION_RESULTS / f"seed{args.seed}" / args.size
        checkpoint = Path("artifacts") / DURATION_RUN / f"seed{args.seed}" / f"{args.size}.pt"
        cache = checkpoint.parent / "predictions" / args.size
    if args.training_epochs == 5:
        allowed_widths = (128, 384, 768) if args.fixed_large_width else (128, BACKBONE_WIDTHS[args.size])
        if args.seconds != 10000 or args.seed != 0 or args.head_width not in allowed_widths:
            parser.error("the width experiment uses 10,000 s, seed 0, and d=128 or d=backbone")
        name = f"{args.size}_d{args.head_width}"
        if args.head_layers == 2:
            name += "_l2"
        if args.ignore_uncertain:
            name += "_ignore_uncertain"
        if args.confidence_targets:
            name += "_confidence"
        if lr_ablation:
            name += "_lr3e-4"
        out = width_results / name
        checkpoint = Path("artifacts") / width_run / f"{name}.pt"
        cache = checkpoint.parent / "predictions" / name
    anchor = json.loads(ANCHOR.read_text())
    reference = anchor["protocol"]
    large = reference["checkpoints"]["self_review"]
    if digest(reference["split"]) != reference["split_sha256"] or digest(large["checkpoint"]) != large["checkpoint_sha256"]:
        raise ValueError("the frozen split or Large checkpoint changed")
    split = json.loads(Path(reference["split"]).read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for partition, key in (("train", "annotations_sha256"), ("validation", "validation_annotations_sha256")):
        file = split["partitions"][partition]["files"]["self_review"]
        if budget and partition == "train":
            file = budget["file"]
        if digest(file["path"]) != file["sha256"] or saved[key] != file["sha256"]:
            raise ValueError(f"unmatched {partition} labels")
    if saved["seed"] != args.seed:
        raise ValueError("checkpoint training seed differs from the requested seed")
    if saved["hidden"] != args.head_width:
        raise ValueError("unexpected predictor width")
    if saved.get("head_layers", 1) != args.head_layers:
        raise ValueError("unexpected predictor depth")
    if saved.get("ignore_uncertain", False) != args.ignore_uncertain:
        raise ValueError("unexpected uncertain-label policy")
    if saved.get("confidence_targets", False) != args.confidence_targets:
        raise ValueError("unexpected BCE target policy")
    if saved["training_config"] != {**large["training_config"], "learning_rate": args.learning_rate}:
        raise ValueError("unmatched optimizer configuration")
    for key in ("target_smoothing", "tv_weight", "height", "width", "dropout",
            "validation_recording_ids"):
        if saved[key] != large[key]:
            raise ValueError(f"unmatched training configuration: {key}")
    expected_ids = budget["recording_ids"] if budget else large["training_recording_ids"]
    expected_windows = budget["windows"] if budget else large["metrics"]["train_windows"]
    if (saved["training_recording_ids"] != expected_ids
            or set(expected_ids) - set(large["training_recording_ids"])
            or saved["metrics"]["train_timebins"] != args.seconds * RATE
            or saved["metrics"]["train_windows"] != expected_windows):
        raise ValueError("unmatched training budget or recording IDs")
    if saved["metrics"]["epochs"] != args.training_epochs or len(saved["training_history"]) != args.training_epochs:
        raise ValueError("unmatched training duration")
    for key in ("validation_timebins", "validation_windows",
            "checkpoint_selection", "threshold_source"):
        if saved["metrics"][key] != large["metrics"][key]:
            raise ValueError(f"unmatched training/checkpoint procedure: {key}")
    if saved["backbone_id"] != f"georgeven/songmae-{args.size}-32x1" or saved["backbone_revision"] != REVISIONS[args.size]:
        raise ValueError("unexpected backbone")
    if (saved["metrics"]["selected_epoch"] != min(saved["training_history"], key=lambda row: row["validation_loss"])["epoch"]
            or saved["target_smoothing"] or saved["tv_weight"] != 0):
        raise ValueError("invalid checkpoint selection or loss recipe")
    duration_pair = None
    if args.training_epochs == 15:
        plan_path = DURATION_RESULTS / "manifest.json"
        plan = json.loads(plan_path.read_text())
        baseline = plan["baselines"][f"{args.size}_seed{args.seed}"]
        if digest(baseline["checkpoint"]) != baseline["checkpoint_sha256"]:
            raise ValueError("three-epoch baseline checkpoint changed")
        previous = torch.load(baseline["checkpoint"], map_location="cpu", weights_only=True)
        if saved["initial_head_sha256"] != previous["initial_head_sha256"]:
            raise ValueError("initial adapter does not match the three-epoch baseline")
        max_loss_difference = 0.0
        for new, old in zip(saved["training_history"][:3], previous["training_history"]):
            for key in ("epoch", "sample_order_sha256", "cpu_rng_sha256"):
                if new[key] != old[key]:
                    raise ValueError(f"three-epoch training prefix changed: {key}")
            for key in ("mean_training_loss", "validation_loss"):
                difference = abs(new[key] - old[key])
                max_loss_difference = max(max_loss_difference, difference)
                if difference > 1e-4:
                    raise ValueError(f"three-epoch loss prefix differs: {key}: {difference}")
        duration_pair = dict(baseline=baseline, manifest_sha256=digest(plan_path),
            matching_initialization_and_rng=True, first_three_epochs_max_loss_difference=max_loss_difference)
    code = ["scripts/evaluate_current_backbones.py", "scripts/evaluate_2d.py", "scripts/evaluate_songmae_smoothing.py",
        "scripts/train.py", "src/birdsong_detect_distill/model.py", "src/birdsong_detect_distill/data.py",
        "src/birdsong_detect_distill/benchmark_data.py", "src/birdsong_detect_distill/benchmark_metrics.py"]
    implementation = reference["code_sha256"]
    if args.head_layers == 2:
        implementation = json.loads((DEPTH_RESULTS / "manifest.json").read_text())["code_sha256"]
    if args.ignore_uncertain:
        implementation = json.loads((UNCERTAIN_RESULTS / "manifest.json").read_text())["code_sha256"]
    if args.confidence_targets:
        implementation = json.loads((CONFIDENCE_RESULTS / "manifest.json").read_text())["code_sha256"]
    if lr_ablation:
        implementation = json.loads((LR_RESULTS / "manifest.json").read_text())["code_sha256"]
    for path in code[1:]:
        if digest(path) != implementation[path]:
            raise ValueError(f"shared training/inference/scoring code differs from Large: {path}")
    for filename, key in (("wav_Files.zip", "audio_sha256"), ("annotation_Files.zip", "references_sha256")):
        if digest(args.root / "powdermill" / filename) != reference[key]:
            raise ValueError("Powdermill audio/references changed")
    metadata = {k: v for k, v in saved.items() if k != "head"}
    metadata.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        trainable_parameters=sum(value.numel() for value in saved["head"].values()))
    protocol = dict(anchor=str(ANCHOR), anchor_sha256=digest(ANCHOR), checkpoint=metadata,
        partitions=reference["partitions"], student_inference=reference["student_inference"],
        coordinate_space=reference["coordinate_space"], mask_rule=reference["mask_rule"],
        threshold_selection=reference["threshold_selection"], threshold_grid=reference["threshold_grid"],
        aggregation=reference["aggregation"], ap_method=reference["ap_method"],
        code_sha256={path: digest(path) for path in code})
    protocol["student_inference"] = {k: v for k, v in reference["student_inference"].items() if k != "visible_devices"}
    if duration_pair:
        protocol["training_duration_pair"] = duration_pair
    if args.training_epochs == 5 and not lr_ablation:
        protocol["predictor_width_experiment"] = dict(backbone_d=BACKBONE_WIDTHS[args.size],
            predictor_d=args.head_width, manifest_sha256=digest(width_results / "manifest.json"))
    if lr_ablation:
        protocol["learning_rate_experiment"] = dict(learning_rate=args.learning_rate, baseline_learning_rate=1e-3,
            manifest_sha256=digest(LR_RESULTS / "manifest.json"))
    if args.head_layers == 2:
        protocol["predictor_depth_experiment"] = dict(layers=2, predictor_d=384,
            manifest_sha256=digest(DEPTH_RESULTS / "manifest.json"))
    if args.ignore_uncertain:
        protocol["uncertain_label_experiment"] = dict(policy="ignore uncertain-only pixels in training and XC validation",
            overlap="target/chorus positive overrides uncertainty", powdermill="all human reference pixels scored as before",
            manifest_sha256=digest(UNCERTAIN_RESULTS / "manifest.json"))
    if args.confidence_targets:
        protocol["confidence_target_experiment"] = dict(policy="BCE target equals maximum foreground-box confidence; zero outside",
            labels=["target_vocalization", "uncertain_vocalization", "chorus"], applied_to="XC training and validation only",
            powdermill="all human reference pixels scored as before", manifest_sha256=digest(CONFIDENCE_RESULTS / "manifest.json"))
    if budget:
        protocol["training_budget"] = budget
        protocol["budget_manifest_sha256"] = digest(SCALING_LABELS / "manifest.json")
    retained_names = [reference["partitions"]["calibration"][0],
        *reference["partitions"]["evaluation"][:2]]
    if args.prediction_cache != "all":
        protocol["prediction_cache"] = dict(policy="samples", retained_segments=retained_names,
            all_segments="exact AP, threshold counts/IoU curves, and raw float32 probability-array hashes retained")
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("cached protocol changed")
    write_json(protocol_path, protocol)
    if (out / "comparison.json").exists():
        print(f"Already complete: {out}")
        return
    score_checkpoint(checkpoint, saved, out, cache, args.root, reference, anchor, args.prediction_cache)


def score_checkpoint(checkpoint, saved, out, cache, root, reference, anchor, prediction_cache="samples"):
    """Shared full-Powdermill scoring; callers freeze and verify their training protocol first."""
    protocol = json.loads((out / "protocol.json").read_text())
    retained_names = [reference["partitions"]["calibration"][0], *reference["partitions"]["evaluation"][:2]]
    cache.mkdir(parents=True, exist_ok=True)
    inventory = {r["name"]: r for r in recordings(root, "powdermill")}
    expected = anchor["per_recording"]["student_self_review"]
    if set(inventory) != set(sum(reference["partitions"].values(), [])):
        raise ValueError("Powdermill partition mismatch")
    device = torch.device("cuda:0")
    backbone = load_backbone(saved["backbone_id"], device, revision=saved["backbone_revision"])
    heads = load_heads([checkpoint], backbone, device)
    scored = {}
    for partition, names in reference["partitions"].items():
        scored[partition] = []
        for index, name in enumerate(names, 1):
            record = inventory[name]
            score_path, path = out / "segments" / f"{name}.json", cache / f"{name}.npz"
            source = dict(audio_sha256=reference["audio_sha256"], member=record["member"],
                reference_sha256=hashlib.sha256(record["events"].tobytes()).hexdigest())
            if score_path.exists():
                item = json.loads(score_path.read_text())
                retained = prediction_cache == "all" or name in retained_names
                if (item["source"] != source or (retained and item["prediction_sha256"] != digest(path))
                        or (not retained and item["prediction_sha256"] is not None)):
                    raise ValueError(f"changed cache: {name}")
            else:
                waveform, rate = read_audio(record)
                duration = len(waveform) / rate
                samples, _ = model_audio(waveform, rate, "songmae")
                width = int(np.ceil(duration * RATE))
                truth, kept = truth_and_coverage(record, width, [[0, duration]])
                if not kept.all():
                    raise ValueError("incomplete scoring interval")
                probability = songmae_probabilities(backbone, heads, samples, device)[checkpoint.stem]
                retained = prediction_cache == "all" or name in retained_names
                if retained:
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, probability=probability)
                    temporary.replace(path)
                value = dict(name=name, group=record["group"], seconds=duration,
                    area={"full": area(smooth(probability)[:, :width], truth)})
                item = dict(source=source, prediction_sha256=digest(path) if retained else None, score=value,
                    prediction_array_sha256=hashlib.sha256(probability.tobytes()).hexdigest())
                write_json(score_path, item)
            value, old = item["score"], expected[partition][index-1]
            tp, _, fn = value["area"]["full"]["counts"][0]
            old_tp, _, old_fn = old["area"]["full"]["counts"][0]
            if any(value[k] != old[k] for k in ("name", "group", "seconds")) or tp+fn != old_tp+old_fn:
                raise ValueError(f"different scoring coverage/reference mask: {name}")
            scored[partition].append(value)
            print(f"{checkpoint.stem} {partition} {index}/{len(names)}: {name}", flush=True)
        if partition == "calibration":
            thresholds = calibrate(scored[partition])
            write_json(out / "calibration.json", dict(threshold_indices=thresholds,
                threshold=float(THRESHOLDS[thresholds["full"]]), segments=names,
                selection=reference["threshold_selection"]))
    report = dict(protocol=protocol, summary=summarize(scored["evaluation"], thresholds)["full"],
        calibration_summary=summarize(scored["calibration"], thresholds)["full"], per_recording=scored,
        evaluated_seconds=sum(row["seconds"] for row in scored["evaluation"]))
    write_json(out / "comparison.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
