#!/usr/bin/env python3
"""Preserve the original XC queue and add fresh, held-out-safe five-second windows."""
import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from audit_overlap import zip_ids
from birdsong_detect_distill.qwen import context, read_done, read_tiles, split_tiles


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-seconds", type=int, default=21000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new continuation directory; never replace a frozen queue")
    base = Path("data/annotations/xcl")
    source = base / "qwen38_agentic_annotations.jsonl"
    master = base / "qwen38_adaptive_review_5s_annotations.jsonl"
    frozen = base / "stage_pipeline_10000train_800val_2026-09-09"
    split = json.loads((frozen / "manifest.json").read_text())
    validation = set(split["partitions"]["validation"]["recording_ids"])
    xcaj = zip_ids(Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw/xcaj/Audio.zip"))
    old = split_tiles(read_tiles(source), 1000)
    if len(old) != len(set(old)) or {tile[0] for tile in old} & xcaj:
        raise ValueError("original queue contains duplicates or XC-AJ recordings")
    done = read_done(master)
    owned = sum(b - a for name, a, b in done if name not in validation | xcaj) / 200
    fresh_count = max(0, math.ceil((args.target_seconds - owned) / 5))
    excluded = validation | xcaj | {tile[0] for tile in old}
    index = Path("data/xcl/shards/index.tsv")
    with index.open() as file:
        candidates = [row for row in csv.DictReader(file, delimiter="\t")
            if row["name"] not in excluded and int(row["end"]) - int(row["start"]) >= 1000]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    chosen, seen = [], set()
    for row in candidates:
        if len(chosen) == fresh_count:
            break
        name, start, end = row["name"], int(row["start"]), int(row["end"])
        if name in seen:
            continue
        seen.add(name)
        owner = rng.randrange((end - start) // 1000) * 1000
        tile = (name, row["shard"], start, end, owner, owner + 1000)
        spec, _, _ = context(Path("data/xcl"), tile, owner, owner + 1000)
        if spec.shape != (128, 1000):
            raise ValueError(f"invalid spectrogram: {tile}")
        chosen.append(tile)
    if len(chosen) != fresh_count:
        raise ValueError("not enough fresh, held-out-safe recordings")
    tiles = old + chosen
    rows = [{"status": "ok", "recording": name, "source": {"shard": shard, "start": start, "end": end},
        "tile": {"start_timebin": a, "end_timebin": b}}
        for name, shard, start, end, a, b in tiles]
    protected = [frozen / "manifest.json", base / "ablation_2500/passes.jsonl",
        Path("results/current_external_2026-09-09/tables.md")]
    protected.extend(frozen.glob("*/*.jsonl"))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "selection": "resume all original tiles; uniformly shuffle fresh recording IDs, one random aligned 5s window each",
        "target_training_eligible_seconds": args.target_seconds,
        "starting_accepted_windows": len(done), "starting_accepted_seconds": sum(b-a for _, a, b in done) / 200,
        "starting_training_eligible_seconds": owned, "fresh_windows": len(chosen), "fresh_seconds": len(chosen)*5,
        "expected_windows": len(tiles), "expected_seconds": sum(t[5]-t[4] for t in tiles) / 200,
        "expected_training_eligible_seconds": sum(t[5]-t[4] for t in tiles if t[0] not in validation) / 200,
        "validation_recordings_excluded_from_training_counts": sorted(validation),
        "xcaj_overlap": [], "source": {"path": str(source), "sha256": digest(source)},
        "index": {"path": str(index), "sha256": digest(index)},
        "original_master_prefix": {"path": str(master), "bytes": master.stat().st_size, "sha256": digest(master)},
        "protected_files": {str(path): digest(path) for path in protected},
        "note": "Existing annotations remain append-only. No training or evaluation is launched. Historical and resumed prompt/token configurations are recorded separately."}
    args.out.mkdir(parents=True)
    queue = args.out / "tiles.jsonl"
    queue.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    report["queue"] = {"path": str(queue), "sha256": digest(queue)}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("protected_files", "validation_recordings_excluded_from_training_counts")}, indent=2))


if __name__ == "__main__":
    main()
