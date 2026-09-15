#!/usr/bin/env python3
"""Freeze reporting intervals and conservatively exclude known pretraining XC IDs."""
import argparse
import json
import re
from pathlib import Path

from birdsong_detect_distill.benchmark_data import digest, recordings, write_json
from birdsong_detect_distill.data import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--teacher-results", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pretraining-index", type=Path, action="append", required=True,
        help="Include both known training and model-selection/validation indices.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("manifest already exists; choose a new path")
    teacher = json.loads(args.teacher_results.read_text())
    coverage = teacher["coverage"]
    reporting_sources = {r["source_recording"] for r in coverage.values()}
    powdermill = list(recordings(args.root, "powdermill"))
    calibration = sorted(r["name"] for r in powdermill if r["group"] not in reporting_sources)
    if not calibration or set(coverage) - {r["name"] for r in powdermill}:
        raise ValueError("invalid Powdermill reporting/calibration partition")
    known, indexes = set(), []
    for path in args.pretraining_index:
        with path.open() as stream:
            next(stream)
            ids = {line.split("\t", 1)[0].upper() for line in stream if line.strip()}
        known |= ids
        indexes.append(dict(path=str(path), sha256=digest(path), recordings=len(ids)))
    rows = read_rows(args.annotations)
    training_ids = {row["recording"].upper() for row in rows}
    xcsl = list(recordings(args.root, "xcsl"))
    identifiers = {}
    for record in xcsl:
        match = re.search(r"XC\d+", record["name"], flags=re.I)
        if not match:
            raise ValueError(f"cannot audit XC recording ID: {record['name']}")
        identifiers[record["name"]] = match.group().upper()
    excluded = {name: id for name, id in identifiers.items() if id in known | training_ids}
    eligible = sorted(set(identifiers) - set(excluded))
    if not eligible:
        raise ValueError("no XC recordings remain after overlap exclusions")
    result = dict(protocol_version=1, teacher_results_sha256=digest(args.teacher_results),
        annotations_sha256=digest(args.annotations), training_windows=len(rows), training_recordings=sorted(training_ids),
        training_seconds=sum((r["tile"]["end_timebin"] - r["tile"]["start_timebin"]) / 200 for r in rows),
        powdermill=dict(role="development_only_also_used_for_BirdCODE_model_selection", calibration=calibration,
            evaluation=coverage, evaluated_seconds=teacher["evaluated_seconds"]),
        xcsl=dict(role="conservative_known_index_disjoint_subset_not_complete_audio_provenance_certification",
            available=len(xcsl), evaluation=eligible, excluded=excluded, indexes=indexes,
            teacher_overlap=sorted(name for name, id in identifiers.items() if id in training_ids)),
        other_datasets=dict(wabad="all recordings from the 68 strongly annotated sites; macro average by site",
            nips4bplus="all strongly annotated clips; birds positive, insect/amphibian/human negative, Unknown intervals ignored"),
        caveat="Excluding known IDs cannot establish absence of duplicates under different IDs or undocumented pretraining exposure.")
    write_json(args.out, result)
    print(json.dumps(dict(calibration_segments=len(calibration), reporting_segments=len(coverage),
        training_windows=len(rows), xcsl_available=len(xcsl), xcsl_excluded=len(excluded), xcsl_eligible=len(eligible)), indent=2))


if __name__ == "__main__":
    main()
