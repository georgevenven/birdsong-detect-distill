#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.evaluation import audio
from birdsong_detect_distill.model import load_detector
from evaluate_2d import THRESHOLDS, iou_curve, reference_mask, songmae_probability
from evaluate_qwen_teacher import references, score_masks, summarize_scores


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen and SongMAE on identical completed Powdermill intervals.")
    parser.add_argument("--teacher-results", type=Path, required=True)
    parser.add_argument("--variant", default="self_review")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "comparison.json").exists():
        raise ValueError("comparison already exists; choose a new output directory")
    teacher_bytes = args.teacher_results.read_bytes()
    teacher = json.loads(teacher_bytes)
    coverage = teacher["coverage"]
    evaluation_sources = {row["source_recording"] for row in coverage.values()}
    source = list(references(args.root))
    calibration = [row for row in source if row[0].split("_Segment_")[0] not in evaluation_sources]
    evaluation = [row for row in source if row[0] in coverage]
    if not calibration or len(evaluation) != len(coverage):
        raise ValueError("need separate calibration sources and references for every evaluation segment")
    device = torch.device(args.device)
    backbone, head, saved = load_detector(args.checkpoint, device)
    args.out.mkdir(parents=True, exist_ok=True)
    per_recording, curves = {}, {}
    with zipfile.ZipFile(args.root / "powdermill/wav_Files.zip") as sounds:
        members = {Path(name).stem: name for name in sounds.namelist() if name.lower().endswith(".wav")}

        def predict(recording, duration, events):
            waveform = audio(sounds, members[recording])
            probability = songmae_probability(backbone, head, waveform, device)[:, :int(np.ceil(duration * 200))]
            return probability, reference_mask(events, probability.shape)

        for index, (recording, duration, events) in enumerate(calibration, 1):
            probability, truth = predict(recording, duration, events)
            curves[recording] = iou_curve(probability, truth).tolist()
            print(f"calibration {index}/{len(calibration)}: {recording}", flush=True)
        mean = np.mean(list(curves.values()), axis=0)
        chosen = int(np.argmax(mean))
        threshold = float(THRESHOLDS[chosen])
        calibration_report = {"selection": "maximum_mean_2d_iou_on_other_source_recordings",
            "threshold": threshold, "mean_iou_2d": float(mean[chosen]), "segments": len(calibration),
            "source_recordings": sorted({row[0].split("_Segment_")[0] for row in calibration}),
            "threshold_grid": THRESHOLDS.tolist(), "per_segment_iou_curves": curves}
        write_json(args.out / "calibration.json", calibration_report)
        print(f"Frozen student probability threshold: {threshold:g}", flush=True)

        for index, (recording, duration, events) in enumerate(evaluation, 1):
            probability, truth = predict(recording, duration, events)
            ownership = np.zeros(probability.shape[1], bool)
            for start, end in coverage[recording]["intervals_seconds"]:
                ownership[round(start * 200):round(end * 200)] = True
            if int(ownership.sum()) != round(coverage[recording]["seconds"] * 200):
                raise ValueError(f"coverage mismatch: {recording}")
            probability, truth = probability[:, ownership], truth[:, ownership]
            qwen = teacher["per_recording"][recording][args.variant]
            for dimension, positives in (("2d", int(truth.sum())), ("temporal", int(truth.any(0).sum()))):
                tp, _, fn = qwen["counts"][dimension]
                if tp + fn != positives:
                    raise ValueError(f"reference mask mismatch: {recording}, {dimension}")
            student = score_masks(probability, probability >= threshold, truth)
            per_recording[recording] = {"qwen": qwen, "songmae": student}
            print(f"comparison {index}/{len(evaluation)}: {recording}, "
                f"2D IoU {qwen['iou_2d']:.3f} -> {student['iou_2d']:.3f}", flush=True)

    models, pooled = {}, {}
    for model in ("qwen", "songmae"):
        models[model], pooled[model] = summarize_scores([row[model] for row in per_recording.values()])
    difference = {key: models["songmae"][key] - value for key, value in models["qwen"].items()
        if value is not None and models["songmae"][key] is not None}
    code_paths = [Path(__file__), Path("scripts/evaluate_qwen_teacher.py"), Path("scripts/evaluate_2d.py"),
        Path("src/birdsong_detect_distill/model.py"), Path("src/birdsong_detect_distill/evaluation.py")]
    result = {"dataset": "powdermill", "comparison": "same_completed_intervals_and_reference_masks",
        "segments": len(evaluation), "source_recordings": sorted(evaluation_sources),
        "evaluated_seconds": teacher["evaluated_seconds"], "coverage": coverage,
        "aggregation": "segment_mean", "ap_method": teacher["ap_method"],
        "empty_conventions": teacher["empty_conventions"], "foreground_labels": teacher["foreground_labels"],
        "coordinate_space": teacher["coordinate_space"], "qwen_variant": args.variant,
        "qwen_operating_point": teacher["operating_point"], "teacher_results": str(args.teacher_results),
        "teacher_results_sha256": hashlib.sha256(teacher_bytes).hexdigest(),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "backbone": saved["backbone_id"], "backbone_revision": getattr(backbone.config, "_commit_hash", None),
        "training": saved["metrics"], "student_probability_threshold": threshold,
        "student_inference": {"window_seconds": 5, "stride_seconds": 2.5, "overlap_aggregation": "maximum",
            "postprocessing": "none", "autocast": "float16", "device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "calibration": {key: value for key, value in calibration_report.items() if key != "per_segment_iou_curves"},
        "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "models": models, "pooled": pooled, "student_minus_teacher": difference, "per_recording": per_recording}
    write_json(args.out / "comparison.json", result)
    print(json.dumps({"models": models, "student_minus_teacher": difference}, indent=2))


if __name__ == "__main__":
    main()
