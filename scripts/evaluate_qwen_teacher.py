#!/usr/bin/env python3
import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import average_precision_score

from birdsong_detect_distill.data import FOREGROUND
from birdsong_detect_distill.evaluation import spans
from evaluate_2d import reference_mask


VARIANTS = ("direct", "reasoning", "self_review", "shifted_review", "full")


def overlap(counts):
    tp, fp, fn = counts
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def measure(probability, prediction, truth, levels):
    counts = [int((prediction & truth).sum()), int((prediction & ~truth).sum()), int((~prediction & truth).sum())]
    ap = None
    if truth.any():
        # Exact pixel AP: group identical confidence scores instead of sorting millions of pixels.
        index = np.searchsorted(levels, probability)
        positive = np.bincount(index[truth], minlength=len(levels))
        total = np.bincount(index.ravel(), minlength=len(levels))
        ap = float(average_precision_score(np.repeat([1, 0], len(levels)), np.tile(levels, 2),
            sample_weight=np.r_[positive, total - positive]))
    return {**overlap(counts), "ap": ap, "counts": counts}


def score_masks(probability, prediction, truth, levels=None):
    levels = np.unique(probability) if levels is None else levels
    spatial = measure(probability, prediction, truth, levels)
    temporal = measure(probability.max(0), prediction.any(0), truth.any(0), levels)
    return {"temporal_iou": temporal["iou"], "iou_2d": spatial["iou"],
        "temporal_precision": temporal["precision"], "temporal_recall": temporal["recall"],
        "area_precision_2d": spatial["precision"], "area_recall_2d": spatial["recall"],
        "temporal_ap": temporal["ap"], "mask_ap_2d": spatial["ap"],
        "counts": {"temporal": temporal["counts"], "2d": spatial["counts"]}}


def summarize_scores(values):
    means = {}
    for metric in values[0]:
        if metric == "counts":
            continue
        defined = [row[metric] for row in values if row[metric] is not None]
        means[metric] = float(np.mean(defined)) if defined else None
    pooled = {dimension: overlap(np.sum([row["counts"][dimension] for row in values], axis=0).tolist())
        for dimension in ("temporal", "2d")}
    return means, pooled


def references(root):
    with zipfile.ZipFile(root / "powdermill/wav_Files.zip") as sounds, zipfile.ZipFile(
            root / "powdermill/annotation_Files.zip") as labels:
        members = {Path(name).stem: name for name in sounds.namelist() if name.lower().endswith(".wav")}
        for member in labels.namelist():
            if not member.lower().endswith(".txt"):
                continue
            recording = Path(member).name.split(".Table")[0]
            with sounds.open(members[recording]) as stream:
                info = sf.info(stream)
            with labels.open(member) as stream:
                table = pd.read_csv(stream, sep="\t")
            columns = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
            yield recording, info.frames / info.samplerate, table[columns].to_numpy(float)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen pipeline stages as union masks on Powdermill.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.annotations.read_bytes()
    rows, error_rows = {}, 0
    for line in snapshot.splitlines():
        row = json.loads(line)
        if row.get("status") == "error":
            error_rows += 1
        if row.get("status") == "ok":
            if set(VARIANTS) - row["variants"].keys():
                raise ValueError("successful window is missing an annotation variant")
            rows[row["recording"], row["owner_start"], row["owner_end"]] = row
    if not rows:
        raise ValueError("no completed annotation windows")
    by_recording = defaultdict(list)
    for row in rows.values():
        by_recording[row["recording"]].append(row)
    scores = {variant: [] for variant in VARIANTS}
    coverage, per_recording = {}, {}
    total_seconds, total_recordings = 0.0, 0
    for recording, duration, events in references(args.root):
        total_seconds += duration
        total_recordings += 1
        if recording not in by_recording:
            continue
        width = int(np.ceil(duration * 200))
        truth = reference_mask(events, (128, width))
        ownership = np.zeros(width, bool)
        for row in by_recording[recording]:
            left, right = max(0, row["owner_start"]), min(width, row["owner_end"])
            ownership[left:right] = True
        if not ownership.any():
            continue
        coverage[recording] = {"seconds": ownership.sum().item() / 200, "duration_seconds": duration,
            "fraction": float(ownership.mean()), "intervals_seconds": spans(ownership, 200).tolist(),
            "source_recording": recording.split("_Segment_")[0]}
        per_recording[recording] = {}
        selected_truth = truth[:, ownership]
        for variant in VARIANTS:
            probability = np.zeros((128, width), np.float32)
            prediction = np.zeros((128, width), bool)
            confidences = [0.0]
            for row in by_recording[recording]:
                for event in row["variants"][variant]["events"]:
                    if event["label"] not in FOREGROUND:
                        continue
                    x0, x1 = max(0, event["start_timebin"]), min(width, event["end_timebin"])
                    y0, y1 = max(0, event["low_mel_bin"]), min(128, event["high_mel_bin"])
                    prediction[y0:y1, x0:x1] = True
                    probability[y0:y1, x0:x1] = np.maximum(probability[y0:y1, x0:x1], event["confidence"])
                    confidences.append(event["confidence"])
            probability = probability[:, ownership]
            prediction = prediction[:, ownership]
            levels = np.unique(np.asarray(confidences, np.float32))
            value = score_masks(probability, prediction, selected_truth, levels)
            scores[variant].append(value)
            per_recording[recording][variant] = value
        print(f"{len(coverage)}/{len(by_recording)}: {recording}, {coverage[recording]['seconds']:g} s", flush=True)
    if missing := set(by_recording) - set(coverage):
        raise ValueError(f"annotation recordings not scored: {sorted(missing)}")
    models, pooled = {}, {}
    for variant, values in scores.items():
        models[variant], pooled[variant] = summarize_scores(values)
    evaluated_seconds = sum(row["seconds"] for row in coverage.values())
    result = {"dataset": "powdermill", "recordings": len(coverage), "dataset_recordings": total_recordings,
        "successful_windows": len(rows), "error_attempts_in_snapshot": error_rows,
        "evaluated_seconds": evaluated_seconds, "dataset_seconds": total_seconds,
        "coverage_fraction": evaluated_seconds / total_seconds, "aggregation": "recording_mean",
        "coordinate_space": "200_hz_time_x_128_mel_frequency", "foreground_labels": sorted(FOREGROUND),
        "operating_point": "union_of_all_emitted_foreground_boxes_without_confidence_filtering",
        "ap_representation": "maximum_label_confidence_inside_boxes_zero_elsewhere",
        "ap_method": "exact_noninterpolated_average_precision; undefined_without_reference_positives",
        "empty_conventions": {"empty_union_iou": 1.0, "undefined_precision_recall": 0.0},
        "annotations": str(args.annotations), "annotations_sha256": hashlib.sha256(snapshot).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": coverage, "models": models, "pooled": pooled, "per_recording": per_recording}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("coverage", "per_recording")}, indent=2))


if __name__ == "__main__":
    main()
