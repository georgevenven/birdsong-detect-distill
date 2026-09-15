#!/usr/bin/env python3
"""Take nested 100 s / 1,000 s prefixes of the frozen, randomly ordered 10,000 s training set."""
import copy
import json
from pathlib import Path

from birdsong_detect_distill.benchmark_data import digest, write_json
from evaluate_current_backbones import ANCHOR, SCALING_LABELS


def main():
    if SCALING_LABELS.exists():
        raise ValueError("budget manifest already exists; preserve the frozen selection")
    anchor = json.loads(ANCHOR.read_text())["protocol"]
    split = json.loads(Path(anchor["split"]).read_text())
    if digest(anchor["split"]) != anchor["split_sha256"]:
        raise ValueError("the parent split changed")
    source = split["partitions"]["train"]["files"]["self_review"]
    validation = split["partitions"]["validation"]["files"]["self_review"]
    for file in (source, validation):
        if digest(file["path"]) != file["sha256"]:
            raise ValueError("the parent labels changed")
    rows = [r for r in map(json.loads, Path(source["path"]).open()) if r.get("status") == "ok"]
    if sum(r["tile"]["end_timebin"] - r["tile"]["start_timebin"] for r in rows) != 2000000:
        raise ValueError("expected the exact 10,000 s training pool")
    report = dict(source_split=anchor["split"], source_split_sha256=anchor["split_sha256"],
        source_annotations=source, validation_annotations=validation,
        selection="nested prefixes of the parent seed-17 randomized window ordering, final window clipped to exact budget",
        validation_seconds=800, budgets={})
    SCALING_LABELS.mkdir(parents=True)
    for seconds in (100, 1000):
        remaining, chosen = seconds * 200, []
        for original in rows:
            if not remaining:
                break
            row = copy.deepcopy(original)
            tile = row["tile"]
            start, end = tile["start_timebin"], tile["end_timebin"]
            end = start + min(remaining, end - start)
            tile.update(end_timebin=end, ownership_end_timebin=end, offset_ms=end * 5, ownership_offset_ms=end * 5)
            chosen.append(row)
            remaining -= end - start
        if remaining:
            raise ValueError("insufficient training duration")
        ids = sorted({r["recording"] for r in chosen})
        if set(ids) & set(split["partitions"]["validation"]["recording_ids"]):
            raise ValueError("validation recording leakage")
        path = SCALING_LABELS / f"self_review_s{seconds}.jsonl"
        metadata = dict(type="metadata", training_seconds=seconds, validation_seconds=800,
            selection=report["selection"], parent_sha256=source["sha256"])
        path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in [metadata, *chosen]))
        report["budgets"][str(seconds)] = dict(seconds=seconds, windows=len(chosen), recording_ids=ids,
            intervals=[[r["recording"], r["tile"]["start_timebin"], r["tile"]["end_timebin"]] for r in chosen],
            file=dict(path=str(path), sha256=digest(path)))
    write_json(SCALING_LABELS / "manifest.json", report)
    print(json.dumps({key: dict(seconds=value["seconds"], windows=value["windows"], recordings=len(value["recording_ids"]))
        for key, value in report["budgets"].items()}, indent=2))


if __name__ == "__main__":
    main()
