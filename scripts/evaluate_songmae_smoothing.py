#!/usr/bin/env python3
"""Matched raw versus probability-smoothed SongMAE masks, with separate development thresholds."""
import argparse
import hashlib
import importlib.metadata
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

from birdsong_detect_distill.benchmark_data import digest, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, THRESHOLDS, area, calibrate, intervals_mask, rasterize, summarize


REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("raw", "gaussian_probability")
SIGMA = (2, 3)


def smooth(probability):
    return ndimage.gaussian_filter(probability, sigma=SIGMA, mode="reflect", truncate=4.0)


def load_prediction(cache, record, audio_hash):
    path = cache / "powdermill" / (record["name"] + ".npz")
    source = json.loads(path.with_suffix(".json").read_text())
    expected = dict(audio_sha256=audio_hash, member=record["member"],
        reference_sha256=hashlib.sha256(record["events"].tobytes()).hexdigest())
    if source["source"] != expected or digest(path) != source["prediction_sha256"]:
        raise ValueError(f"changed cached predictions or source: {record['name']}")
    with np.load(path) as data:
        probability = data["probability"]
    width = int(np.ceil(source["duration"] * RATE))
    if probability.shape[0] != 128 or probability.shape[1] < width or not np.isfinite(probability).all():
        raise ValueError("invalid full-segment prediction cache")
    return probability, source, width


def truth_and_coverage(record, width, intervals):
    truth = rasterize(np.column_stack((record["events"], np.ones(len(record["events"])))), width).astype(bool)
    return truth, intervals_mask(intervals, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/baselines_matched_2026-09-07/paper/songmae"))
    parser.add_argument("--manifest", type=Path, default=Path("results/baselines_matched_2026-09-07/manifest.json"))
    parser.add_argument("--baseline", type=Path, default=Path("results/baselines_matched_2026-09-07/powdermill_final/songmae.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() or args.workers < 1:
        raise ValueError("choose a new output directory and positive worker count")
    manifest = json.loads(args.manifest.read_text())
    baseline = json.loads(args.baseline.read_text())
    metadata = json.loads((args.cache / "metadata.json").read_text())
    if metadata != baseline["inference"] or digest(args.manifest) != baseline["manifest_sha256"]:
        raise ValueError("model or coverage differs from the existing matched comparison")
    if digest(metadata["checkpoint"]) != metadata["checkpoint_sha256"]:
        raise ValueError("changed checkpoint")
    for path, expected in metadata["code_sha256"].items():
        if digest(REPO / path) != expected:
            raise ValueError(f"changed inference code: {path}")
    code_paths = [Path(__file__).relative_to(REPO), Path("src/birdsong_detect_distill/benchmark_data.py"),
        Path("src/birdsong_detect_distill/benchmark_metrics.py")]
    code = {str(path): digest(REPO / path) for path in code_paths}
    audio_hash = digest(args.root / "powdermill/wav_Files.zip")
    powdermill = manifest["powdermill"]
    selected = {"calibration": set(powdermill["calibration"]), "evaluation": set(powdermill["evaluation"])}
    groups = {part: {name.split("_Segment_")[0] for name in names} for part, names in selected.items()}
    if groups != {"calibration": {"Recording_2", "Recording_3", "Recording_4"}, "evaluation": {"Recording_1"}}:
        raise ValueError("unexpected or overlapping source-recording split")
    inventory = {row["name"]: row for row in recordings(args.root, "powdermill")}
    if (selected["calibration"] | selected["evaluation"]) - inventory.keys():
        raise ValueError("missing recordings")
    old = {part: {row["name"]: row for row in rows} for part, rows in baseline["per_recording"].items()}
    args.out.mkdir(parents=True)
    scored = {condition: {part: [] for part in selected} for condition in CONDITIONS}
    sources, thresholds = {}, {}

    def score(name, partition):
        record = inventory[name]
        probability, source, width = load_prediction(args.cache, record, audio_hash)
        intervals = powdermill["evaluation"][name]["intervals_seconds"] if partition == "evaluation" else [[0, source["duration"]]]
        truth, kept = truth_and_coverage(record, width, intervals)
        result = {}
        for condition in CONDITIONS:
            # Preserve full spectrogram context, including the cached STFT endpoint frame, before scoring crops.
            values = probability if condition == "raw" else smooth(probability)
            score = area(values[:, :width][:, kept], truth[:, kept])
            result[condition] = dict(name=name, group=record["group"], seconds=float(kept.sum() / RATE), area={"full": score})
        if result["raw"]["area"]["full"] != old[partition][name]["area"]["full"]:
            raise ValueError(f"raw scores no longer reproduce the baseline: {name}")
        return name, result, source

    for partition, names in selected.items():
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = pool.map(lambda name: score(name, partition), sorted(names))
            for index, (name, results, source) in enumerate(jobs, 1):
                for condition, row in results.items():
                    scored[condition][partition].append(row)
                sources[name] = source
                print(f"{partition} {index}/{len(names)} {name}", flush=True)
        if partition == "calibration":
            thresholds = {condition: calibrate(scored[condition][partition]) for condition in CONDITIONS}
            write_json(args.out / "calibration.json", dict(selection="maximum mean segment 2D IoU on Recordings 2–4 only",
                threshold_grid=THRESHOLDS.tolist(), threshold_indices=thresholds,
                thresholds={key: float(THRESHOLDS[value['full']]) for key, value in thresholds.items()},
                per_recording={key: rows[partition] for key, rows in scored.items()}))
            print(f"Thresholds frozen before reporting-set scoring: {thresholds}", flush=True)
    summaries = {condition: summarize(rows["evaluation"], thresholds[condition])["full"] for condition, rows in scored.items()}
    if summaries["raw"] != baseline["summary"]["full"]:
        raise ValueError("raw aggregate result changed")
    seconds = sum(row["seconds"] for row in scored["raw"]["evaluation"])
    if seconds != powdermill["evaluated_seconds"]:
        raise ValueError("matched reporting coverage changed")
    report = dict(dataset="powdermill", role="development comparison, not an unseen test", inference=metadata,
        manifest=str(args.manifest), manifest_sha256=digest(args.manifest), baseline=str(args.baseline),
        baseline_sha256=digest(args.baseline), raw_baseline_reproduction="exact per-recording AP, counts, IoU curves and summary",
        smoothing=dict(space="probability", sigma_mel_time=SIGMA, frame_ms=5, mode="reflect", truncate=4.0,
            dtype="float32", scope="full cached spectrogram before duration/interval/excerpt cropping"),
        binary_postprocessing="single threshold only; no hysteresis, closing, component filtering, or hole filling",
        threshold_selection="independent maximum mean 2D IoU on the same 41 calibration segments; frozen before reporting scoring",
        threshold_indices=thresholds, mask_rule="float64 comparison of stored float32 scores to the selected float64 threshold",
        area_ap_method="exact noninterpolated pixel AP from continuous raw or smoothed scores, before thresholding",
        aggregation="segment-mean AP/IoU; pooled pixel precision/recall", empty_union_iou=1,
        coordinate_space="128 Slaney mel bins, 20–16000 Hz, 200 Hz time grid", calibration_segments=len(selected["calibration"]),
        reporting_segments=len(selected["evaluation"]), evaluated_seconds=seconds, summary=summaries,
        smoothed_minus_raw={key: summaries["gaussian_probability"][key] - summaries["raw"][key]
            for key in ("ap", "iou", "pooled_precision", "pooled_recall")},
        calibration_summary={condition: summarize(rows["calibration"], thresholds[condition])["full"]
            for condition, rows in scored.items()},
        per_recording=scored, sources=sources, code_sha256=code,
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy")})
    write_json(args.out / "comparison.json", report)
    print(json.dumps({"summary": summaries, "smoothed_minus_raw": report["smoothed_minus_raw"]}, indent=2))


if __name__ == "__main__":
    main()
