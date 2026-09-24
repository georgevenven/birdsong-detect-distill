#!/usr/bin/env python3
import argparse
import json
import re
import zipfile
from pathlib import Path


XC = re.compile(r"XC\d+", re.IGNORECASE)


def index_ids(path):
    with path.open() as file:
        next(file)
        return {line.split("\t", 1)[0].upper() for line in file if line.strip()}


def zip_ids(path):
    with zipfile.ZipFile(path) as archive:
        return {match.group().upper() for name in archive.namelist() if (match := XC.search(name))}


def annotation_ids(path):
    return {row["recording"].upper() for row in (json.loads(line) for line in path.open()) if row.get("status") == "ok"}


def main():
    parser = argparse.ArgumentParser(description="Audit XCL/Qwen recording overlap with held-out BirdCODE corpora.")
    parser.add_argument("--xcl-index", type=Path, default=Path("data/xcl/shards/index.tsv"))
    parser.add_argument("--xcl-val-index", type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
        default=Path("data/annotations/xcl/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--birdcode-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/data_overlap.json"))
    args = parser.parse_args()

    train, validation = index_ids(args.xcl_index), index_ids(args.xcl_val_index)
    labeled = annotation_ids(args.annotations)
    xcaj_audio = args.birdcode_root / "xcaj" / "Audio.zip"
    with zipfile.ZipFile(xcaj_audio) as archive:
        xcaj = {split: {match.group().upper() for name in archive.namelist()
            if name.startswith(split + "/") and (match := XC.search(name))}
            for split in ("Training", "Validation")}
    powdermill = zip_ids(args.birdcode_root / "powdermill" / "wav_Files.zip")
    wabad = set().union(*(zip_ids(path) for path in (args.birdcode_root / "wabad").glob("*.zip")))

    def overlap(left, right):
        values = sorted(left & right)
        return {"count": len(values), "recordings": values}

    report = {
        "xcl": {"train_recordings": len(train), "validation_recordings": len(validation),
            "train_validation_overlap": overlap(train, validation)},
        "qwen": {"recordings": len(labeled),
            "xcaj_training_overlap": overlap(labeled, xcaj["Training"]),
            "xcaj_validation_overlap": overlap(labeled, xcaj["Validation"]),
            "powdermill_xc_id_overlap": overlap(labeled, powdermill),
            "wabad_xc_id_overlap": overlap(labeled, wabad)},
        "full_xcl_train": {
            "xcaj_training_overlap": overlap(train, xcaj["Training"]),
            "xcaj_validation_overlap": overlap(train, xcaj["Validation"])},
        "held_out": {"xcaj_training_recordings": len(xcaj["Training"]),
            "xcaj_validation_recordings": len(xcaj["Validation"]),
            "powdermill_xc_ids": len(powdermill), "wabad_xc_ids": len(wabad)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
