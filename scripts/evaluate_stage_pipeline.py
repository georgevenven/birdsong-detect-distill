#!/usr/bin/env python3
"""Calibrate teacher/student IoU on Powdermill 2–4, then report complete Recording_1."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, THRESHOLDS, area, calibrate, summarize
from birdsong_detect_distill.data import FOREGROUND
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import load_heads, songmae_probabilities
from evaluate_songmae_smoothing import smooth, truth_and_coverage
from prepare_stage_split import VARIANTS


LABELS = ("Initial prediction (with reasoning)", "+ Self-review",
    "+ Shifted-context review and reconciliation", "+ Conditional adjudication")


def teacher_probability(rows, variant, width):
    probability = np.zeros((128, width), np.float32)
    for row in rows:
        for event in row["variants"][variant]["events"]:
            if event["label"] in FOREGROUND:
                x0, x1 = max(0, event["start_timebin"]), min(width, event["end_timebin"])
                y0, y1 = max(0, event["low_mel_bin"]), min(128, event["high_mel_bin"])
                probability[y0:y1, x0:x1] = np.maximum(probability[y0:y1, x0:x1], event["confidence"])
    return probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--teacher-results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "comparison.json").exists():
        raise ValueError("completed comparison exists; choose a new output directory")
    split = json.loads(args.split.read_text())
    teacher = json.loads(args.teacher_results.read_text())
    if teacher["coverage_fraction"] != 1 or teacher["recordings"] != 77:
        raise ValueError("complete Powdermill teacher coverage is required")
    master = Path(teacher["annotations"])
    if digest(master) != teacher["annotations_sha256"]:
        raise ValueError("teacher annotations changed after their evaluation")
    by_recording = defaultdict(list)
    seen = set()
    for row in map(json.loads, master.open()):
        if row.get("status") != "ok":
            continue
        key = (row["recording"], row["owner_start"], row["owner_end"])
        if key in seen:
            raise ValueError(f"duplicate teacher window: {key}")
        seen.add(key)
        by_recording[row["recording"]].append(row)
    metadata, paths = {}, []
    for variant in VARIANTS:
        path = args.checkpoints / f"{variant}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        train, val = split["partitions"]["train"], split["partitions"]["validation"]
        if (saved["target_smoothing"] or saved["tv_weight"] != 0
                or saved["annotations_sha256"] != train["files"][variant]["sha256"]
                or saved["validation_annotations_sha256"] != val["files"][variant]["sha256"]
                or saved["training_recording_ids"] != train["recording_ids"]
                or saved["validation_recording_ids"] != val["recording_ids"]
                or saved["metrics"]["train_timebins"] != 10000 * RATE
                or saved["metrics"]["validation_timebins"] != 800 * RATE
                or saved["metrics"]["epochs"] != 3
                or saved["metrics"]["threshold_source"] != "external_calibration_required"):
            raise ValueError(f"incompatible data or training recipe: {variant}")
        metadata[variant] = {k: v for k, v in saved.items() if k != "head"}
        metadata[variant].update(checkpoint=str(path), checkpoint_sha256=digest(path))
        paths.append(path)
    first = metadata[VARIANTS[0]]
    for variant, saved in metadata.items():
        for key in ("backbone_id", "backbone_revision", "seed", "initial_head_sha256", "training_config"):
            if saved[key] != first[key]:
                raise ValueError(f"unmatched {key}: {variant}")
        if [r["sample_order_sha256"] for r in saved["training_history"]] != [
                r["sample_order_sha256"] for r in first["training_history"]]:
            raise ValueError(f"unmatched training window order: {variant}")
    records = {r["name"]: r for r in recordings(args.root, "powdermill")}
    if set(records) != set(by_recording) or len(seen) != teacher["successful_windows"]:
        raise ValueError("teacher/reference coverage mismatch")
    partitions = {"calibration": sorted(n for n, r in records.items() if r["group"] != "Recording_1"),
        "evaluation": sorted(n for n, r in records.items() if r["group"] == "Recording_1")}
    if len(partitions["calibration"]) != 41 or len(partitions["evaluation"]) != 36:
        raise ValueError("unexpected Powdermill source-recording split")
    code = ["scripts/evaluate_stage_pipeline.py", "scripts/evaluate_2d.py", "scripts/evaluate_songmae_smoothing.py",
        "scripts/train.py", "src/birdsong_detect_distill/model.py", "src/birdsong_detect_distill/data.py",
        "src/birdsong_detect_distill/benchmark_data.py", "src/birdsong_detect_distill/benchmark_metrics.py"]
    protocol = dict(split=str(args.split), split_sha256=digest(args.split), checkpoints=metadata,
        teacher_results=str(args.teacher_results), teacher_results_sha256=digest(args.teacher_results),
        teacher_annotations_sha256=digest(master),
        audio_sha256=digest(args.root / "powdermill/wav_Files.zip"),
        references_sha256=digest(args.root / "powdermill/annotation_Files.zip"),
        partitions=partitions, code_sha256={p: digest(p) for p in code},
        coordinate_space="128 Slaney mel bins (20–16000 Hz), 200 Hz time grid",
        student_inference=dict(window_seconds=5, stride_seconds=2.5, overlap_aggregation="maximum",
            smoothing="probability", sigma_mel_time=[2, 3], mode="reflect", truncate=4.0,
            binary_postprocessing="single calibrated threshold only", autocast="float16",
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
        teacher_inference="maximum foreground-box confidence per pixel, zero outside; no smoothing",
        mask_rule="stored float32 scores compared as float64 to the selected float64 threshold",
        threshold_selection="per-model maximum segment-mean 2D IoU on Recordings_2–4, frozen before Recording_1",
        threshold_grid=THRESHOLDS.tolist(), aggregation="segment_mean",
        ap_method="exact noninterpolated pixel AP; omit segments without reference positives",
        empty_union_iou=1)
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("existing evaluation protocol differs; choose new output/cache directories")
    write_json(protocol_path, protocol)
    device = torch.device("cuda:0")
    backbone = load_backbone(first["backbone_id"], device, revision=first["backbone_revision"])
    heads = load_heads(paths, backbone, device)
    models = [f"{kind}_{variant}" for kind in ("teacher", "student") for variant in VARIANTS]
    scored = {model: {part: [] for part in partitions} for model in models}
    for partition, names in partitions.items():
        for index, name in enumerate(names, 1):
            path = args.cache / f"{name}.npz"
            score_path = args.out / "segments" / f"{name}.json"
            if score_path.exists():
                item = json.loads(score_path.read_text())
                if item["prediction_sha256"] != digest(path):
                    raise ValueError(f"cached prediction changed: {name}")
            else:
                record = records[name]
                waveform, rate = read_audio(record)
                duration = len(waveform) / rate
                samples, _ = model_audio(waveform, rate, "songmae")
                width = int(np.ceil(duration * RATE))
                truth, kept = truth_and_coverage(record, width, [[0, duration]])
                if not kept.all() or teacher["coverage"][name]["seconds"] != duration:
                    raise ValueError(f"incomplete scoring interval: {name}")
                predictions = songmae_probabilities(backbone, heads, samples, device)
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **predictions)
                temporary.replace(path)
                values = {}
                for variant in VARIANTS:
                    probability = teacher_probability(by_recording[name], variant, width)
                    teacher_score = area(probability, truth)
                    old = teacher["per_recording"][name][variant]
                    tp, _, fn = old["counts"]["2d"]
                    if tp + fn != int(truth.sum()) or (teacher_score["ap"] is None) != (old["mask_ap_2d"] is None):
                        raise ValueError(f"teacher reference mismatch: {name}")
                    if teacher_score["ap"] is not None and not np.isclose(teacher_score["ap"], old["mask_ap_2d"], rtol=0, atol=1e-12):
                        raise ValueError(f"teacher AP no longer reproduces: {name}, {variant}")
                    student_score = area(smooth(predictions[variant])[:, :width], truth)
                    for kind, score in (("teacher", teacher_score), ("student", student_score)):
                        values[f"{kind}_{variant}"] = dict(name=name, group=record["group"], seconds=duration,
                            area={"full": score})
                item = dict(prediction_sha256=digest(path), models=values)
                write_json(score_path, item)
            for model, value in item["models"].items():
                scored[model][partition].append(value)
            print(f"{partition} {index}/{len(names)}: {name}", flush=True)
        if partition == "calibration":
            thresholds = {model: calibrate(rows[partition]) for model, rows in scored.items()}
            write_json(args.out / "calibration.json", dict(threshold_indices=thresholds,
                thresholds={model: float(THRESHOLDS[t["full"]]) for model, t in thresholds.items()},
                segments=names, selection=protocol["threshold_selection"]))
            print(f"Frozen calibration indices: {thresholds}", flush=True)
    summary = {model: summarize(rows["evaluation"], thresholds[model])["full"] for model, rows in scored.items()}
    write_json(args.out / "comparison.json", dict(protocol=protocol, summary=summary,
        calibration_summary={model: summarize(rows["calibration"], thresholds[model])["full"]
            for model, rows in scored.items()}, per_recording=scored,
        evaluated_seconds=sum(r["seconds"] for r in scored[models[0]]["evaluation"])))
    lines = ["# Calibrated teacher–student stage comparison", "",
        "| VLM annotation pipeline (cumulative) | Teacher pixel AP | Teacher 2D IoU | Student pixel AP | Student 2D IoU |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for label, variant in zip(LABELS, VARIANTS):
        t, s = summary[f"teacher_{variant}"], summary[f"student_{variant}"]
        lines.append(f"| {label} | {t['ap']:.3f} | {t['iou']:.3f} | {s['ap']:.3f} | {s['iou']:.3f} |")
    lines.extend(["", "Each student uses frozen SongMAE-Large, identical 10,000 s XC training audio, hard targets and plain BCE.",
        "The checkpoint with minimum BCE on 800 s of recording-disjoint XC validation audio is selected from three epochs.",
        "Student probabilities are Gaussian-smoothed (2 mel bins, 3 frames/15 ms), then thresholded once; no other cleanup.",
        "Each teacher/student threshold maximizes mean 2D IoU on the same 41 segments from Recordings_2–4.",
        "All scores report complete Recording_1: 36 segments, 10,800 s; AP averages its 35 positive segments.",
        "AP uses continuous scores before thresholding. Teacher scores are maximum box confidences, without smoothing.",
        "Shifted review merges agreeing proposals and retains self-review on disagreement; the full pipeline adjudicates disagreements.",
        "The historical direct baseline is excluded because its no-reasoning setting was not reliably enforced.", ""])
    (args.out / "pipeline_table.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
