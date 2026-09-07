#!/usr/bin/env python3
import argparse
import copy
import json
import random
from pathlib import Path


VARIANTS = ("direct", "reasoning", "self_review", "shifted_review", "full")


def main():
    parser = argparse.ArgumentParser(description="Create matched nested time-budget subsets from a paused Qwen ablation.")
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, nargs="+", default=(100, 500, 1000, 4000))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = {}
    for row in (json.loads(line) for line in args.master.open()):
        if row.get("status") == "ok" and row["owner_end"] > row["owner_start"]:
            rows[(row["recording"], row["owner_start"], row["owner_end"])] = row
    keys = sorted(rows)
    random.Random(args.seed).shuffle(keys)
    available_bins = sum(rows[key]["owner_end"] - rows[key]["owner_start"] for key in keys)
    if available_bins < max(args.seconds) * 200:
        raise ValueError(f"need {max(args.seconds)} seconds, found {available_bins / 200:g}")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.master), "seed": args.seed,
        "available_seconds": available_bins / 200, "subsets": {}}
    for seconds in sorted(args.seconds):
        remaining, chosen = seconds * 200, []
        for key in keys:
            row = rows[key]
            size = min(remaining, row["owner_end"] - row["owner_start"])
            chosen.append((row, row["owner_start"], row["owner_start"] + size))
            remaining -= size
            if not remaining:
                break
        directory = args.out / f"{seconds}s"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"type": "metadata", "workflow": "qwen_ablation_time_subset", "seconds": seconds,
            "windows": len(chosen), "seed": args.seed}
        for variant in VARIANTS:
            output = [metadata]
            for row, start, end in chosen:
                annotation = copy.deepcopy(row["variants"][variant])
                annotation["tile"]["start_timebin"] = start
                annotation["tile"]["end_timebin"] = end
                annotation["tile"]["ownership_start_timebin"] = start
                annotation["tile"]["ownership_end_timebin"] = end
                output.append(annotation)
            (directory / f"{variant}.jsonl").write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output))
        report["subsets"][str(seconds)] = {"windows": len(chosen),
            "recordings": len({row["recording"] for row, _, _ in chosen})}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
