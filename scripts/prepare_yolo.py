#!/usr/bin/env python3
import argparse
import hashlib
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
    if not 0 < end - start <= 1000:
        raise ValueError("YOLO teacher windows must contain at most five seconds")
    return np.pad(spec, ((0, 0), (valid_start - start, 1000 - (valid_end - start))), constant_values=-100)


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
            # Match SongMAE's rasterized teacher targets, including shortened tiles.
            start, end = row["tile"]["start_timebin"], row["tile"]["end_timebin"]
            left, right = max(start, event["start_timebin"]) - start, min(end, event["end_timebin"]) - start
            top, bottom = (128 - event["high_mel_bin"]) / 128 * 1000, (128 - event["low_mel_bin"]) / 128 * 1000
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
    parser.add_argument("--validation-annotations", type=Path, help="Explicit recording-disjoint XC validation labels.")
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/yolo/qwen"))
    parser.add_argument("--val-fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-all", action="store_true", help="Use the complete supplied label budget; no held-out validation.")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new output directory; existing datasets are never overwritten")
    if args.validation_annotations and args.train_all:
        parser.error("explicit validation cannot be combined with --train-all")
    rows = read_rows(args.annotations, args.seed)
    if args.validation_annotations:
        train, val = rows, read_rows(args.validation_annotations, args.seed)
        if not train or not val or {r['recording'].upper() for r in train} & {r['recording'].upper() for r in val}:
            raise ValueError("explicit training/validation must be nonempty and recording-disjoint")
    else:
        train, val = (rows, []) if args.train_all else split_rows(rows, args.val_fraction, args.seed)
    summary = {name: write_split(rows, name, args.out, args.shard_dir)
        for name, rows in (("train", train), ("val", val))}
    args.out.mkdir(parents=True, exist_ok=True)
    val_path = "images/train" if args.train_all else "images/val"
    (args.out / "dataset.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: {val_path}\nnames:\n  0: bird\n")
    summary.update(annotations=str(args.annotations), annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        train_all=args.train_all, seed=args.seed, image_representation="5s_128_mel_viridis_2048x512",
        training_seconds=sum((r["tile"]["end_timebin"]-r["tile"]["start_timebin"])/200 for r in train),
        training_recordings=sorted({r["recording"] for r in train}),
        validation_annotations=str(args.validation_annotations) if args.validation_annotations else None,
        validation_annotations_sha256=hashlib.sha256(args.validation_annotations.read_bytes()).hexdigest() if args.validation_annotations else None,
        validation_seconds=sum((r["tile"]["end_timebin"]-r["tile"]["start_timebin"])/200 for r in val),
        validation_recordings=sorted({r["recording"] for r in val}),
        validation_note="train alias required by YOLO; not held-out and not used for checkpoint selection" if args.train_all else "recording-disjoint")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
