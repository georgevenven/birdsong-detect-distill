#!/usr/bin/env python3
"""Paper-ready area/frame tables; SongMAE external cells deliberately remain pending."""
import csv
import json
from pathlib import Path

from birdsong_detect_distill.benchmark_data import digest, write_json


RESULTS = Path("results/competitors_external_2026-09-08")
LABELS = {"birdbox": "YOLO11n (released)", "qwen_yolo": "YOLO11n (our teacher labels)", "birdcode": "BirdCODE"}
METRICS = ("ap", "iou", "pooled_precision", "pooled_recall")


def main():
    if (RESULTS / "tables.md").exists():
        raise ValueError("paper tables already exist")
    manifest = Path("results/baselines_matched_2026-09-07/manifest.json")
    reports, provenance = {}, {}
    for model in LABELS:
        datasets = ("xcsl", "nips4bplus") if model == "birdcode" else ("wabad", "hawaii", "xcsl", "nips4bplus")
        calibration = json.loads(Path(f"results/baselines_matched_2026-09-07/powdermill_final/{model}.json").read_text())
        inference = calibration["inference"]
        if digest(inference["checkpoint"]) != inference["checkpoint_sha256"]:
            raise ValueError(f"competitor checkpoint changed: {model}")
        for source, checksum in inference["code_sha256"].items():
            if digest(source) != checksum:
                raise ValueError(f"native inference code changed: {source}")
        for dataset in datasets:
            path = RESULTS / f"{model}_{dataset}.json"
            r = json.loads(path.read_text())
            if (r["diagnostic_only"] or r["inference"] != calibration["inference"] or r["dataset"] != dataset
                    or r["manifest_sha256"] != digest(manifest) or r["threshold_indices"] != calibration["threshold_indices"]
                    or r["calibration_sha256"] != digest(f"results/baselines_matched_2026-09-07/powdermill_final/{model}.json")):
                raise ValueError(f"non-final or mismatched competitor result: {path}")
            if r["segments"] != {"wabad": 4264, "hawaii": 635, "xcsl": 288, "nips4bplus": 674}[dataset]:
                raise ValueError(f"unexpected primary coverage: {path}")
            for source, checksum in r["scoring_code_sha256"].items():
                if digest(source) != checksum:
                    raise ValueError(f"scoring code changed: {source}")
            reports[model, dataset] = r
            provenance[path.name] = digest(path)
    for dataset in ("wabad", "hawaii", "xcsl", "nips4bplus"):
        runs = [r for (model, name), r in reports.items() if name == dataset]
        reference = runs[0]
        for r in runs[1:]:
            if ([x["name"] for x in r["per_recording"]["evaluation"]] != [x["name"] for x in reference["per_recording"]["evaluation"]]
                    or r["evaluated_seconds"] != reference["evaluated_seconds"] or r["source_audio_sha256"] != reference["source_audio_sha256"]):
                raise ValueError(f"competitors use different recordings/audio: {dataset}")
            for actual, expected in zip(r["per_recording"]["evaluation"], reference["per_recording"]["evaluation"]):
                for metric in actual["area"].keys() & expected["area"].keys():
                    a, e = actual["area"][metric], expected["area"][metric]
                    if ([tp + fn for tp, fp, fn in a["counts"]] != [tp + fn for tp, fp, fn in e["counts"]]
                            or actual["seconds"] != expected["seconds"]):
                        raise ValueError(f"competitors use different reference masks: {dataset}/{actual['name']}/{metric}")
    def row(model, datasets, metric):
        values = []
        for dataset in datasets:
            report = reports[model, dataset]
            summary = report["site_macro" if dataset == "wabad" else "summary"][metric]
            values.extend(summary[key] for key in METRICS)
        return [LABELS[model], *values]
    specs = {
        "two_dimensional": (["WABAD", "Hawaii"], ["Pixel AP", "2D IoU", "Pixel precision", "Pixel recall"],
            [row(model, ("wabad", "hawaii"), "full") for model in ("birdbox", "qwen_yolo")]),
        "temporal": (["XC-AJ", "NIPS4Bplus"], ["Frame AP", "Temporal IoU", "Frame precision", "Frame recall"],
            [row(model, ("xcsl", "nips4bplus"), "temporal") for model in ("birdcode", "birdbox", "qwen_yolo")])}
    lines = ["# External competitor tables", "", "SongMAE external evaluation is intentionally deferred while the detector is developed.", ""]
    for name, (datasets, metrics, rows) in specs.items():
        header = ["Model", *[f"{d} {m}" for d in datasets for m in metrics]]
        rows.append(["SongMAE-Large (ours)", *["Pending"] * 8])
        for suffix, delimiter in ((".csv", ","), (".tsv", "\t")):
            with (RESULTS / (name + suffix)).open("w") as stream:
                writer = csv.writer(stream, delimiter=delimiter)
                writer.writerow(header)
                writer.writerows(rows)
        lines += [f"## {'Time–frequency localization' if name == 'two_dimensional' else 'Temporal occupancy'}", "",
            "| " + " | ".join(header) + " |", "|---|" + "---:|" * 8]
        lines += ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else v for v in values) + " |" for values in rows]
        lines += [""]
    lines += ["## Protocol notes", "",
        "WABAD: 4,264 primary recordings, 68 sites. Three of the 4,267 candidates contain inverted annotation bounds and are excluded uniformly; zero-extent references contribute zero area. "
        "AP/IoU use within-site recording means followed by equal site weighting; precision/recall pool pixels within each site, then average sites equally. "
        "Hawaii: 635 original FLAC recordings; AP/IoU are recording means and precision/recall pool pixels. Full 20–16,000 Hz, 128-bin mel area is primary; common-band and per-site alternatives remain in the complete reports.", "",
        "XC-AJ: the frozen 288-recording known-index-disjoint subset, not the complete 967-recording dataset. NIPS4Bplus: 674 annotated clips, non-birds negative, Unknown intervals ignored. "
        "Frame AP/IoU are recording means; frame precision/recall pool counts. All models use the same 200-Hz scoring grid, without changing their native input or output resolution.", "",
        "All operating thresholds are fixed from Powdermill Recordings 2–4. Released YOLO: pixel 0.05, frame 0.04; teacher-trained YOLO: 0.01 for both; BirdCODE: frame 0.20. "
        "AP uses continuous foreground scores. Empty-union IoU is one; AP excludes recordings without positive reference area. These are box-derived area/occupancy metrics, not box AP@0.5 or event F1.", "",
        "The Hawaii CSV contains 196 zero-duration boxes; they remain in the source but contribute zero reference area. Faint/unidentified calls may be unannotated. "
        "Broader pretraining exposure is not fully certified; see [Hawaii's original release](https://zenodo.org/records/7078499) and the [BirdBox paper](https://arxiv.org/html/2606.10407v1).", "",
        "CSV/TSV files retain full precision. `table_sources.json` records the report checksums. Historical results and the manuscript's older teacher-stage student values are not overwritten."]
    (RESULTS / "tables.md").write_text("\n".join(lines) + "\n")
    write_json(RESULTS / "table_sources.json", dict(reports_sha256=provenance, manifest_sha256=digest(manifest),
        table_generator_sha256=digest(__file__), songmae_external_evaluation="not performed",
        cross_model_checks="identical recording names, audio hashes, scoring seconds and positive reference counts; frozen native inference and calibration"))
    keys = ("dataset", "inference", "segments", "evaluated_seconds", "summary", "site_macro",
        "threshold_indices", "calibration_sha256", "reference_annotation_policy", "dataset_protocol")
    write_json(RESULTS / "summary.json", {f"{model}_{dataset}": {k: r[k] for k in keys if k in r}
        for (model, dataset), r in reports.items()})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
