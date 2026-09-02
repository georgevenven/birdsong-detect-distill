#!/usr/bin/env python3
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from birdsong_detect_distill.data import FOREGROUND, load_spec_slice, read_rows, split_rows
from birdsong_detect_distill.qwen import image


def spectrogram(row, shard_dir):
    source, tile = row["source"], row["tile"]
    start, end = tile["start_timebin"], tile["end_timebin"]
    valid_start, valid_end = max(0, start), min(source["end"] - source["start"], end)
    spec = load_spec_slice(shard_dir / Path(source["shard"]).name,
        source["start"] + valid_start, source["start"] + valid_end)
    return np.pad(spec, ((0, 0), (valid_start - start, end - valid_end)), constant_values=-100)


def write_split(rows, name, root, shard_dir):
    images, labels = root / "images" / name, root / "labels" / name
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    def write(item):
        index, row = item
        stem = f"{row['recording']}_{row['tile']['ownership_start_timebin']:07d}_{index:05d}"
        image(spectrogram(row, shard_dir)).save(images / f"{stem}.png")
        lines = []
        for event in row["events"]:
            if event["label"] not in FOREGROUND:
                continue
            left, top, right, bottom = event["bbox_2d"]
            if left < right and top < bottom:
                lines.append(f"0 {(left + right) / 2000:.6f} {(top + bottom) / 2000:.6f} "
                    f"{(right - left) / 1000:.6f} {(bottom - top) / 1000:.6f}")
        (labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        return len(lines)
    with ThreadPoolExecutor(8) as pool:
        boxes = sum(pool.map(write, enumerate(rows)))
    return {"windows": len(rows), "boxes": boxes, "recordings": len({x["recording"] for x in rows})}


def main():
    parser = argparse.ArgumentParser(description="Render reviewed Qwen boxes as a recording-disjoint YOLO dataset.")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/xcl/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/yolo/qwen"))
    parser.add_argument("--val-fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train, val = split_rows(read_rows(args.annotations, args.seed), args.val_fraction, args.seed)
    summary = {name: write_split(rows, name, args.out, args.shard_dir)
        for name, rows in (("train", train), ("val", val))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
