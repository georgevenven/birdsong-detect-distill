#!/usr/bin/env python3
"""Freeze exact-duration, recording-disjoint XC partitions shared by the four teacher stages."""
import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from audit_overlap import zip_ids


VARIANTS = ("reasoning", "self_review", "shifted_review", "full")
RATE = 200


def validation_recordings(durations, target, maximum, seed):
    names = sorted(durations)
    random.Random(seed).shuffle(names)
    reachable, history = 1, []
    mask = (1 << (maximum + 1)) - 1
    for name in names:
        history.append(reachable)
        reachable = (reachable | reachable << durations[name]) & mask
    above = reachable >> target
    if not above:
        raise ValueError("cannot allocate recording-disjoint validation without reducing training duration")
    remaining = target + (above & -above).bit_length() - 1
    chosen = set()
    for name, before in reversed(list(zip(names, history))):
        if not (before >> remaining) & 1:
            chosen.add(name)
            remaining -= durations[name]
    if remaining:
        raise ValueError("validation allocation did not reconstruct")
    return chosen


def select(rows, budget, seed):
    keys = sorted(rows)
    random.Random(seed).shuffle(keys)
    chosen = []
    for key in keys:
        row = rows[key]
        start = row["owner_start"]
        size = min(budget, row["owner_end"] - start)
        chosen.append((row, start, start + size))
        budget -= size
        if not budget:
            return chosen
    raise ValueError("insufficient annotation duration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--xcaj-audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-seconds", type=int, default=10000)
    parser.add_argument("--validation-seconds", type=int, default=800)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.out.exists() or min(args.train_seconds, args.validation_seconds) <= 0:
        raise ValueError("choose a new output directory and positive durations")
    snapshot = args.master.read_bytes()
    rows = {(r["recording"], r["owner_start"], r["owner_end"]): r
        for r in map(json.loads, snapshot.splitlines()) if r.get("status") == "ok"}
    durations, by_recording = defaultdict(int), defaultdict(list)
    for key, row in rows.items():
        name, start, end = key
        if not name.upper().startswith("XC") or end <= start or set(VARIANTS) - row["variants"].keys():
            raise ValueError(f"invalid or incomplete XC window: {key}")
        base = row["variants"]["reasoning"]
        for variant in VARIANTS:
            item = row["variants"][variant]
            if (item["source"] != base["source"] or item["recording"] != name
                    or item["tile"]["ownership_start_timebin"] != start
                    or item["tile"]["ownership_end_timebin"] != end
                    or not 0 <= start < end <= item["source"]["end"] - item["source"]["start"]):
                raise ValueError(f"variant/source mismatch: {key}, {variant}")
        durations[name] += end - start
        by_recording[name].append((start, end))
    for name, intervals in by_recording.items():
        intervals.sort()
        if any(b > c for (_, b), (c, _) in zip(intervals, intervals[1:])):
            raise ValueError(f"overlapping source windows: {name}")
    overlap = {name.upper() for name in durations} & zip_ids(args.xcaj_audio)
    if overlap:
        raise ValueError(f"XC-AJ leakage: {sorted(overlap)}")
    total = sum(durations.values())
    train_bins, val_bins = args.train_seconds * RATE, args.validation_seconds * RATE
    if total < train_bins + val_bins:
        raise ValueError(f"need {args.train_seconds + args.validation_seconds}s; found {total / RATE:g}s")
    val_ids = validation_recordings(durations, val_bins, total - train_bins, args.seed)
    partitions = {
        "train": select({k: r for k, r in rows.items() if k[0] not in val_ids}, train_bins, args.seed),
        "validation": select({k: r for k, r in rows.items() if k[0] in val_ids}, val_bins, args.seed),
    }
    report = dict(source=str(args.master), source_sha256=hashlib.sha256(snapshot).hexdigest(), seed=args.seed,
        rate=RATE, available_seconds=total / RATE, available_recordings=len(durations),
        unused_seconds=(total - train_bins - val_bins) / RATE, xcaj_overlap=[],
        selection="seeded recording-level subset sum for validation; seeded windows clipped to exact duration",
        partitions={})
    args.out.mkdir(parents=True)
    for partition, chosen in partitions.items():
        directory = args.out / partition
        directory.mkdir()
        seconds = sum(end - start for _, start, end in chosen) / RATE
        ids = sorted({r["recording"] for r, _, _ in chosen})
        manifest = dict(seconds=seconds, windows=len(chosen), recording_ids=ids, files={},
            intervals=[[r["recording"], start, end] for r, start, end in chosen])
        for variant in VARIANTS:
            output = [dict(type="metadata", workflow="recording_disjoint_stage_ablation", variant=variant,
                partition=partition, seconds=seconds, windows=len(chosen), seed=args.seed)]
            for row, start, end in chosen:
                item = copy.deepcopy(row["variants"][variant])
                item["teacher_view"] = item["tile"].copy()
                item["tile"].update(start_timebin=start, end_timebin=end,
                    ownership_start_timebin=start, ownership_end_timebin=end,
                    onset_ms=start * 5, offset_ms=end * 5,
                    ownership_onset_ms=start * 5, ownership_offset_ms=end * 5)
                output.append(item)
            path = directory / f"{variant}.jsonl"
            path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in output))
            manifest["files"][variant] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report["partitions"][partition] = manifest
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**{k: v for k, v in report.items() if k != "partitions"},
        "partitions": {k: {f: v[f] for f in ("seconds", "windows", "recording_ids")}
            for k, v in report["partitions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
