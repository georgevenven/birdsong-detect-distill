"""Original Hawaii FLAC release; zero-duration annotations have no reference area."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_data import digest


def protocol(root):
    root = Path(root)
    source = json.loads((root / "manifest.json").read_text())
    if source["distribution"] != "zenodo" or source["recordings"] != 635 or source["annotations"] != 59583:
        raise ValueError("expected the original complete Hawaii release")
    if digest(root / "raw/annotations.csv") != source["annotation_sha256"]:
        raise ValueError("Hawaii annotation source changed")
    return dict(source=source["source"], manifest_sha256=digest(root / "manifest.json"),
        annotation_sha256=source["annotation_sha256"], recordings=635, raw_boxes=59583,
        zero_duration_boxes=196, positive_area_boxes=59387, hours=source["hours"],
        annotation_policy="Zero-duration boxes contribute zero reference area; do not expand them through grid rounding. Original CSV is unchanged.",
        primary_aggregation="recording-mean AP/IoU; pooled pixel precision/recall",
        limitations="Faint/unidentifiable calls may be unannotated; pretraining provenance is not fully certified.")


def recordings(root):
    root = Path(root)
    details = protocol(root)
    source = json.loads((root / "manifest.json").read_text())
    table = pd.read_csv(root / "raw/annotations.csv")
    inventory = {r["filename"]: r for r in source["audio"]}
    if set(table.Filename) != set(inventory) or {p.name for p in (root / "audio").glob("*.flac")} != set(inventory):
        raise ValueError("Hawaii recording inventory changed")
    columns = ["Start Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
    if int((table[columns[0]] == table[columns[1]]).sum()) != details["zero_duration_boxes"]:
        raise ValueError("unexpected zero-duration annotation count")
    for filename, rows in table.groupby("Filename", sort=True):
        events = rows[columns].to_numpy(float)
        info = inventory[filename]
        if (not np.isfinite(events).all() or (events[:, 0] < 0).any()
                or (events[:, 1] < events[:, 0]).any() or (events[:, 1] > info["seconds"] + .02).any()
                or (events[:, 2] < 0).any() or (events[:, 3] <= events[:, 2]).any() or (events[:, 3] > 16000).any()):
            raise ValueError(f"invalid Hawaii annotation: {filename}")
        yield dict(name=Path(filename).stem, path=str(root / "audio" / filename),
            events=events[events[:, 1] > events[:, 0]], group=filename.split("_")[2],
            expected_audio_sha256=info["sha256"], zero_duration_boxes=int((events[:, 1] == events[:, 0]).sum()))
