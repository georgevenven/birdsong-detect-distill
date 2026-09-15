#!/usr/bin/env python3
"""Prepare the existing two-panel plot inputs from the matched current experiments."""
import csv
import json

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from evaluate_current_backbones import ANCHOR, RESULTS, SCALING_LABELS, SCALING_RESULTS


def main():
    anchor = json.loads(ANCHOR.read_text())
    backbone_path = RESULTS / "summary.json"
    backbones = json.loads(backbone_path.read_text())
    for path, sha in backbones["sources_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"backbone comparison source changed: {path}")
    models = {}
    for name, row in backbones["models"].items():
        if digest(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError(f"backbone checkpoint changed: {name}")
        models[name] = dict(training_seconds=10000, checkpoint=row["checkpoint"],
            checkpoint_sha256=row["checkpoint_sha256"], mask_ap_2d=row["ap"], iou_2d=row["iou"])
    sources = {str(ANCHOR): digest(ANCHOR), str(backbone_path): digest(backbone_path)}
    students = []
    for seconds in (100, 1000):
        path = SCALING_RESULTS / f"s{seconds}" / "comparison.json"
        run = json.loads(path.read_text())
        if (run["protocol"]["anchor_sha256"] != digest(ANCHOR)
                or run["protocol"]["partitions"] != anchor["protocol"]["partitions"]
                or run["protocol"]["budget_manifest_sha256"] != digest(SCALING_LABELS / "manifest.json")):
            raise ValueError(f"unmatched budget experiment: {seconds}")
        rows = run["per_recording"]
        measured = summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"]
        saved = run["protocol"]["checkpoint"]
        if (measured != run["summary"] or run["evaluated_seconds"] != 10800
                or digest(saved["checkpoint"]) != saved["checkpoint_sha256"]
                or saved["metrics"]["train_timebins"] != seconds * 200):
            raise ValueError(f"budget score/checkpoint does not reproduce: {seconds}")
        students.append(dict(training_seconds=seconds, checkpoint=saved["checkpoint"],
            checkpoint_sha256=saved["checkpoint_sha256"], mask_ap_2d=measured["ap"], iou_2d=measured["iou"],
            threshold=measured["threshold"], training_windows=saved["metrics"]["train_windows"],
            selected_epoch=saved["metrics"]["selected_epoch"]))
        sources[str(path)] = digest(path)
    students.append(models["large"])
    shared = dict(dataset="powdermill", evaluated_seconds=10800, segments=36,
        source_recordings=["Recording_1"], aggregation="segment_mean",
        qwen_self_review=dict(mask_ap_2d=anchor["summary"]["teacher_self_review"]["ap"],
            iou_2d=anchor["summary"]["teacher_self_review"]["iou"]),
        validation_seconds=800, sources_sha256=sources)
    write_json(SCALING_RESULTS / "scaling.json", dict(**shared, students=students))
    write_json(SCALING_RESULTS / "backbones.json", dict(**shared, training_seconds=10000, models=models))
    with (SCALING_RESULTS / "figure_values.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["Panel", "Configuration", "Mask AP", "2D IoU"])
        for row in students:
            writer.writerow(["a", f"{row['training_seconds']} s", row["mask_ap_2d"], row["iou_2d"]])
        for name, row in models.items():
            writer.writerow(["b", name.title(), row["mask_ap_2d"], row["iou_2d"]])
    caption = ("Figure 3. Effect of (a) training-label budget for SongMAE-Large and (b) backbone size at 10,000 s "
        "of training audio. All students use self-reviewed Qwen labels, hard-target BCE training, and Gaussian-smoothed "
        "predictions. Training subsets are nested; checkpoint selection uses the same separate 800 s of recording-disjoint "
        "XC validation audio. Mask AP is evaluated on the 35 positive segments of complete Powdermill Recording_1. "
        "The dashed line denotes the self-reviewed Qwen teacher. Each configuration uses one training seed.\n")
    (SCALING_RESULTS / "caption.txt").write_text(caption)
    print(json.dumps(dict(training_budget_ap=[row["mask_ap_2d"] for row in students],
        backbone_ap={name: row["mask_ap_2d"] for name, row in models.items()}), indent=2))


if __name__ == "__main__":
    main()
