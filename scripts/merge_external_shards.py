#!/usr/bin/env python3
"""Combine complete, disjoint external recording shards; never tune operating thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from birdsong_detect_distill import hawaii
from birdsong_detect_distill.benchmark_data import digest, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import area_scores, summarize


def reference_policy(rows, inventory, model, dataset):
    """Do not turn malformed annotated recordings into apparent negatives."""
    selected = {row["name"] for row in rows}
    excluded, zero_extent, negative_onsets = {}, {}, 0
    records = {r["name"]: r for r in inventory if r["name"] in selected}
    for name, record in records.items():
        e = record["events"]
        inverse = (e[:, 1] < e[:, 0]) | (e[:, 3] < e[:, 2])
        zero = (e[:, 1] == e[:, 0]) | (e[:, 3] == e[:, 2])
        negative_onsets += int((e[:, 0] < 0).sum())
        if inverse.any():
            excluded[name] = dict(inverted_bounds_boxes=int(inverse.sum()), raw_boxes=len(e))
        elif zero.any():
            zero_extent[name] = int(zero.sum())
    corrected = []
    for row in rows:
        name = row["name"]
        if name in excluded:
            continue
        if name in zero_extent:
            from birdsong_detect_distill.baseline_models import maps
            path = Path("artifacts/competitors_external_2026-09-08") / model / dataset / (name.replace("/", "__") + ".npz")
            cached = json.loads(path.with_suffix(".json").read_text())
            e = records[name]["events"]
            if (digest(path) != cached["prediction_sha256"]
                    or hashlib.sha256(e.tobytes()).hexdigest() != cached["source"]["reference_sha256"]):
                raise ValueError("changed native predictions or raw references during label audit")
            with np.load(path) as data:
                probability, score = maps({key: data[key] for key in data.files}, cached["duration"])
            valid = (e[:, 1] > e[:, 0]) & (e[:, 3] > e[:, 2])
            row = {**row, "area": area_scores(probability, score, e[valid], len(score), [[0, cached["duration"]]],
                records[name].get("ignored", []), records[name].get("temporal_only", False))}
        corrected.append(row)
    return corrected, dict(excluded_recordings=excluded, zero_extent_boxes_by_retained_recording=zero_extent,
        negative_onsets_clipped_to_recording_start=negative_onsets, candidate_recordings=len(rows),
        primary_recordings=len(corrected),
        policy="Exclude recordings with inverted annotation bounds uniformly; zero-extent reference boxes contribute no area before grid rounding. Source annotations and native predictions are unchanged.")


def by_site(rows, thresholds):
    groups = sorted({r["group"] for r in rows})
    sites = {g: summarize([r for r in rows if r["group"] == g], thresholds) for g in groups}
    def mean_defined(values):
        values = [x for x in values if x is not None]
        return float(np.mean(values)) if values else None
    macro = {metric: {key: mean_defined([sites[g][metric][key] for g in groups])
        for key in ("ap", "iou", "pooled_precision", "pooled_recall")} for metric in next(iter(sites.values()))}
    return sites, macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("birdbox", "qwen_yolo", "birdcode"), required=True)
    parser.add_argument("--dataset", choices=("wabad", "hawaii", "xcsl", "nips4bplus"), required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    out = Path("results/competitors_external_2026-09-08")
    target = out / f"{args.model}_{args.dataset}.json"
    if target.exists():
        raise ValueError("final report already exists")
    paths = [out / "parts" / f"{args.model}_{args.dataset}_{i}.json" for i in range(args.shards)]
    parts = [json.loads(path.read_text()) for path in paths]
    first = parts[0]
    for i, part in enumerate(parts):
        if part["sharding"]["index"] != i or part["sharding"]["shards"] != args.shards:
            raise ValueError("incorrect shard identities")
        for key in ("inference", "manifest_sha256", "dataset", "threshold_indices", "event_metrics", "scoring_code_sha256"):
            if part[key] != first[key]:
                raise ValueError(f"incompatible shards: {key}")
        if part["dataset"] != args.dataset or part["inference"]["model"] != args.model or part["event_metrics"]:
            raise ValueError("expected matching area-only competitor results")
    manifest_path = Path("results/baselines_matched_2026-09-07/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    calibration_path = Path(f"results/baselines_matched_2026-09-07/powdermill_final/{args.model}.json")
    calibration = json.loads(calibration_path.read_text())
    if (digest(manifest_path) != first["manifest_sha256"] or calibration["inference"] != first["inference"]
            or calibration["threshold_indices"] != first["threshold_indices"]):
        raise ValueError("changed frozen calibration")
    root = Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw")
    if args.dataset == "hawaii":
        inventory = list(hawaii.recordings("data/hawaii/zenodo"))
    else:
        inventory = list(recordings(Path("data/nips4bplus") if args.dataset == "nips4bplus" else root, args.dataset))
    expected = set(manifest["xcsl"]["evaluation"]) if args.dataset == "xcsl" else {r["name"] for r in inventory}
    rows = sorted([row for part in parts for row in part["per_recording"]["evaluation"]], key=lambda r: r["name"])
    if [row["name"] for row in rows] != sorted(expected):
        raise ValueError("missing/duplicated external recording coverage")
    raw_summary = summarize(rows, first["threshold_indices"])
    raw_site_macro = by_site(rows, first["threshold_indices"])[1] if args.dataset in {"wabad", "hawaii"} else None
    rows, annotation_policy = reference_policy(rows, inventory, args.model, args.dataset)
    sources = {}
    for part in parts:
        for path, checksum in part["source_audio_sha256"].items():
            if path in sources and sources[path] != checksum:
                raise ValueError("inconsistent audio source hashes")
            sources[path] = checksum
    report = {key: value for key, value in first.items() if key not in ("per_site", "site_macro", "sharding")}
    report.update(diagnostic_only=False, per_recording={"evaluation": rows}, segments=len(rows),
        evaluated_seconds=sum(r["seconds"] for r in rows), source_audio_sha256=sources,
        summary=summarize(rows, first["threshold_indices"]),
        reference_annotation_policy=annotation_policy,
        uncorrected_annotation_diagnostic=dict(summary=raw_summary, site_macro=raw_site_macro,
            warning="Includes malformed source annotations; not the primary paper result"),
        merged_shards={str(p): digest(p) for p in paths}, calibration_sha256=digest(calibration_path),
        coverage_verification="all eligible recording names exactly once; no external calibration")
    if args.dataset in {"wabad", "hawaii"}:
        report["per_site"], report["site_macro"] = by_site(rows, first["threshold_indices"])
    report["merge_code_sha256"] = digest(__file__)
    write_json(target, report)
    print(json.dumps(dict(model=args.model, dataset=args.dataset, segments=len(rows), summary=report["summary"],
        site_macro=report.get("site_macro")), indent=2))


if __name__ == "__main__":
    main()
