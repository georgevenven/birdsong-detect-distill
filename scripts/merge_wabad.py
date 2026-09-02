#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from birdsong_detect_distill.evaluation import metrics


def main():
    parser = argparse.ArgumentParser(description="Merge parallel WABAD evaluation shards.")
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/reproduced/songmae_large_32x1_wabad.json"))
    args = parser.parse_args()
    documents = [json.loads(path.read_text()) for path in args.parts]
    counts = [np.load(path.with_suffix(".npz")) for path in args.parts]
    frame = sum((x["frame"] for x in counts), np.zeros_like(counts[0]["frame"]))
    event = {key: sum((x[name] for x in counts), np.zeros_like(counts[0][name]))
        for key, name in ((.2, "event_02"), (.5, "event_05"))}
    per_site = {site: score for document in documents for site, score in document["per_site"].items()}
    result = {"model": documents[0]["model"], "dataset": "wabad", "task": "binary_any_bird_detection",
        "files": sum(x["files"] for x in documents), **metrics(frame, event),
        "site_macro": {key: float(np.mean([score[key] for score in per_site.values()])) for key in next(iter(per_site.values()))},
        "sites": len(per_site), "per_site": per_site}
    result.update({key: documents[0][key] for key in ("fine_tuned", "confidence_floor", "initialization", "teacher")
        if key in documents[0]})
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(args.out.with_suffix(".npz"), frame=frame, event_02=event[.2], event_05=event[.5])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
