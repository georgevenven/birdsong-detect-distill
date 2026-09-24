#!/usr/bin/env python3
"""Export the XCL spectrogram windows used by the paper into small shards readable by PixelWindows.

Each original shard contributes only the rows referenced by data/labels/*.jsonl, in the same int8
format with its companion affine .txt. Label files are rewritten with remapped `source.start`
offsets (data/labels_subset/), so training code runs unchanged with `--shards artifacts/data/xcl_subset`.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--shards", type=Path, required=True, help="original BirdSet XCL shard directory")
parser.add_argument("--labels", type=Path, default=REPO / "data/labels")
parser.add_argument("--out", type=Path, default=REPO / "artifacts/data/xcl_subset")
parser.add_argument("--remapped", type=Path, default=REPO / "data/labels_subset")
args = parser.parse_args()

rows = {path: [json.loads(line) for line in path.read_text().splitlines()] for path in sorted(args.labels.glob("*.jsonl"))}
spans = defaultdict(set)
for items in rows.values():
    for row in items:
        source, tile = row["source"], row["tile"]
        spans[source["shard"]].add((source["start"] + tile["start_timebin"], source["start"] + tile["end_timebin"]))

args.out.mkdir(parents=True, exist_ok=True)
offset = {}
for shard, ranges in sorted(spans.items()):
    array = np.load(args.shards / shard, mmap_mode="r")
    pieces, position = [], 0
    for start, end in sorted(ranges):
        offset[shard, start] = position - start
        pieces.append(np.asarray(array[start:end]))
        position += end - start
    np.save(args.out / shard, np.concatenate(pieces))
    affine = (args.shards / shard).with_suffix(".txt")
    if affine.exists():
        shutil.copy2(affine, (args.out / shard).with_suffix(".txt"))

args.remapped.mkdir(parents=True, exist_ok=True)
for path, items in rows.items():
    with open(args.remapped / path.name, "w") as out:
        for row in items:
            source, tile = row["source"], row["tile"]
            shift = offset[source["shard"], source["start"] + tile["start_timebin"]]
            remapped = dict(row, source=dict(source, start=source["start"] + shift, end=source["end"] + shift))
            out.write(json.dumps(remapped, separators=(",", ":")) + "\n")
print(f"{len(spans)} shards, {sum(len(v) for v in spans.values())} windows -> {args.out}")
