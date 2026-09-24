#!/usr/bin/env python3
"""Native model inputs, matched recording coverage, shared area / optional event metrics."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from birdsong_detect_distill.baseline_models import Predictor, maps, native_events
from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, area_scores, calibrate, event_counts, intervals_mask, summarize
from birdsong_detect_distill.evaluation import spans
from birdsong_detect_distill import hawaii

REPO = Path(__file__).resolve().parents[1]
INFERENCE_FILES = ["src/birdsong_detect_distill/baseline_models.py", "src/birdsong_detect_distill/birdbox.py",
    "src/birdsong_detect_distill/benchmark_data.py", "src/birdsong_detect_distill/model.py",
    "scripts/evaluate_2d.py", "scripts/evaluate_qwen_yolo.py"]


def main(default_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("birdbox", "qwen_yolo", "birdcode", "songmae"), default=default_model, required=default_model is None)
    parser.add_argument("--dataset", choices=("powdermill", "xcsl", "wabad", "nips4bplus", "hawaii"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yolo11n", "yolo11l"), default="yolo11n")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=.00001)
    parser.add_argument("--max-det", type=int, default=10000)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="This model's complete Powdermill report; required on external datasets.")
    parser.add_argument("--event-metrics", action="store_true", help="Also score native events (more expensive).")
    parser.add_argument("--score-only", action="store_true", help="Use frozen predictions without loading a GPU model.")
    parser.add_argument("--maximum", type=int, help="Diagnostic subset per partition; cannot calibrate a final result.")
    parser.add_argument("--shards", type=int, default=1, help="Disjoint external recording shards; merge before reporting.")
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new output file; historical results are never overwritten")
    if not 0 < args.confidence <= .001 or args.batch_size < 1 or args.max_det < 1:
        raise ValueError("AP requires a small positive confidence floor and positive batch/max-det sizes")
    if args.shards < 1 or not 0 <= args.shard_index < args.shards or (args.shards > 1 and args.dataset == "powdermill"):
        raise ValueError("sharding requires a valid external recording partition")
    manifest = json.loads(args.manifest.read_text())
    if args.dataset != "powdermill" and args.calibration is None:
        raise ValueError("external evaluation requires development-selected thresholds, never test-set tuning")
    inventory = list(hawaii.recordings(args.root) if args.dataset == "hawaii" else recordings(args.root, args.dataset))
    if args.dataset == "powdermill":
        development = set(manifest["powdermill"]["calibration"])
        reporting = set(manifest["powdermill"]["evaluation"])
        if development & reporting:
            raise ValueError("calibration and reporting recordings overlap")
        if {x.split("_Segment_")[0] for x in development} & {x.split("_Segment_")[0] for x in reporting}:
            raise ValueError("calibration and reporting original source recordings overlap")
        partitions = {"calibration": [r for r in inventory if r["name"] in development],
            "evaluation": [r for r in inventory if r["name"] in reporting]}
        if len(partitions["calibration"]) != len(development) or len(partitions["evaluation"]) != len(reporting):
            raise ValueError("missing manifest recordings")
    else:
        selected = set(manifest["xcsl"]["evaluation"]) if args.dataset == "xcsl" else {r["name"] for r in inventory}
        partitions = {"evaluation": [r for r in inventory if r["name"] in selected]}
        if len(partitions["evaluation"]) != len(selected):
            raise ValueError("missing manifest recordings")
    expected_recordings = {part: len(rows) for part, rows in partitions.items()}
    if args.shards > 1:
        partitions = {part: rows[args.shard_index::args.shards] for part, rows in partitions.items()}
    if args.maximum:
        partitions = {k: v[:args.maximum] for k, v in partitions.items()}
    args.cache.mkdir(parents=True, exist_ok=True)
    metadata_path = args.cache / "metadata.json"
    code = {p: digest(REPO / p) for p in INFERENCE_FILES}
    if args.score_only:
        metadata = json.loads(metadata_path.read_text())
        if metadata["model"] != args.model or metadata["code_sha256"] != code:
            raise ValueError("cached model or inference code differs; regenerate in a new cache")
        predictor = None
    else:
        predictor = Predictor(args.model, args.checkpoint, args.device, args.batch_size, args.confidence,
            args.max_det, args.variant, args.backbone_revision)
        metadata = {**predictor.metadata, "code_sha256": code}
        with (args.cache / "metadata.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
                raise ValueError("cache configuration differs; choose a new cache")
            if not metadata_path.exists():
                write_json(metadata_path, metadata)
    if args.model == "qwen_yolo":
        training = metadata.get("training") or {}
        dataset = training.get("dataset", {})
        if dataset.get("annotations_sha256") != manifest["annotations_sha256"] or not dataset.get("train_all"):
            raise ValueError("YOLO checkpoint does not use the manifest's complete teacher-label budget")
        if training.get("sha256") != metadata["checkpoint_sha256"]:
            raise ValueError("YOLO checkpoint sidecar checksum mismatch")
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if calibration:
        if calibration["diagnostic_only"] or calibration["dataset"] != "powdermill":
            raise ValueError("calibration must be a complete Powdermill development result")
        if calibration["inference"] != metadata or calibration["manifest_sha256"] != digest(args.manifest):
            raise ValueError("calibration and evaluation model/protocol differ")
        if calibration["event_metrics"] != args.event_metrics:
            raise ValueError("calibration must include the same metrics as evaluation")
    source_hashes = {}
    scored = {name: [] for name in partitions}
    for partition, selected in partitions.items():
        for index, record in enumerate(selected, 1):
            name = record["name"]
            audio_source = record.get("archive", record.get("path"))
            if audio_source not in source_hashes:
                source_hashes[audio_source] = digest(audio_source)
            if record.get("expected_audio_sha256", source_hashes[audio_source]) != source_hashes[audio_source]:
                raise ValueError(f"audio differs from the original release: {name}")
            signature = dict(audio_sha256=source_hashes[audio_source], member=record.get("member"),
                reference_sha256=hashlib.sha256(record["events"].tobytes()).hexdigest())
            path = args.cache / args.dataset / (name.replace("/", "__") + ".npz")
            record_meta = path.with_suffix(".json")
            if path.exists():
                saved = json.loads(record_meta.read_text())
                if saved["source"] != signature or saved["prediction_sha256"] != digest(path):
                    raise ValueError(f"cached input/prediction changed: {name}")
                with np.load(path) as archive:
                    prediction = {key: archive[key] for key in archive.files}
                duration = saved["duration"]
            else:
                if predictor is None:
                    raise ValueError(f"missing cached prediction: {name}")
                audio, sample_rate = read_audio(record)
                duration = len(audio) / sample_rate
                audio, sample_rate = model_audio(audio, sample_rate, args.model)
                prediction = predictor.predict(audio, sample_rate)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **prediction)
                write_json(record_meta, dict(source=signature, duration=duration, prediction_sha256=digest(path)))
            intervals = manifest["powdermill"]["evaluation"][name]["intervals_seconds"] if args.dataset == "powdermill" and partition == "evaluation" else [[0, duration]]
            probability, score = maps(prediction, duration)
            ignored = record.get("ignored", [])
            width = len(score)
            kept = intervals_mask(intervals, width, ignored)
            row = dict(name=name, group=record["group"], seconds=float(kept.sum() / RATE),
                area=area_scores(probability, score, record["events"], width, intervals, ignored, record.get("temporal_only", False)))
            if args.event_metrics:
                valid_intervals = spans(kept, RATE)
                row["events"] = event_counts(native_events(prediction, probability), record["events"][:, :2], valid_intervals)
            scored[partition].append(row)
            print(f"{partition} {index}/{len(selected)} {name}: temporal AP={row['area']['temporal']['ap']}", flush=True)
    thresholds = calibration["threshold_indices"] if calibration else calibrate(scored["calibration"])
    report = dict(protocol_version=1, dataset=args.dataset, diagnostic_only=args.maximum is not None or args.shards > 1,
        sharding=dict(index=args.shard_index, shards=args.shards, expected_recordings=expected_recordings),
        role=manifest["powdermill"]["role"] if args.dataset == "powdermill" else manifest["xcsl"]["role"] if args.dataset == "xcsl" else "external_evaluation_pretraining_provenance_not_certified",
        inference=metadata, manifest_sha256=digest(args.manifest), source_audio_sha256=source_hashes,
        event_metrics=args.event_metrics, event_ap_method="101-threshold interpolated PR area; native event proposals; class-agnostic maximum-cardinality matching",
        area_ap_method="exact_noninterpolated_unique_score_weighted_AP", area_aggregation="segment_mean; AP excludes segments without positive reference area",
        coordinate_space="200_Hz_time_x_128_Slaney_mel_bins_20_to_16000_Hz; common band selects intersecting 500_to_12000_Hz rows",
        empty_union_iou=1, threshold_indices=thresholds, threshold_selection="separate development-mean area IoU optima; event F1 from pooled calibration counts",
        evaluated_seconds=sum(r["seconds"] for r in scored["evaluation"]), segments=len(scored["evaluation"]),
        summary=summarize(scored["evaluation"], thresholds), per_recording=scored,
        scoring_code_sha256={p: digest(REPO / p) for p in ["scripts/evaluate_baselines.py", "src/birdsong_detect_distill/benchmark_metrics.py",
            "src/birdsong_detect_distill/evaluation.py", "src/birdsong_detect_distill/hawaii.py"]})
    if args.dataset == "powdermill" and not args.maximum and report["evaluated_seconds"] != manifest["powdermill"]["evaluated_seconds"]:
        raise ValueError("scoring coverage differs from matched teacher/student intervals")
    if args.dataset == "hawaii":
        report["dataset_protocol"] = hawaii.protocol(args.root)
    if args.dataset in {"wabad", "hawaii"}:
        groups = sorted({r["group"] for r in scored["evaluation"]})
        report["per_site"] = {g: summarize([r for r in scored["evaluation"] if r["group"] == g], thresholds) for g in groups}
        def defined_mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None
        report["site_macro"] = {metric: {key: defined_mean([report["per_site"][g][metric][key] for g in groups])
            for key in ("ap", "iou", "pooled_precision", "pooled_recall")}
            for metric in report["summary"] if metric != "events"}
        if args.event_metrics:
            report["site_macro"]["events"] = {iou: {key: defined_mean([report["per_site"][g]["events"][iou][key] for g in groups])
                for key in ("ap", "f1")} for iou in ("0.2", "0.5")}
    write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
