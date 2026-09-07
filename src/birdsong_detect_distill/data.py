import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FOREGROUND = {"target_vocalization", "uncertain_vocalization", "chorus"}


def load_spec_slice(path, start, end):
    path = Path(path)
    array = np.load(path, mmap_mode="r")[start:end]
    if array.dtype == np.int8:
        affine = np.atleast_2d(np.loadtxt(path.with_suffix(".txt"), dtype=np.float32))
        return np.array((array.astype(np.float32) * affine[None, :, 0] + affine[None, :, 1]).T)
    return np.array(array.T, dtype=np.float32, copy=True)


def read_rows(path, seed=0, maximum=None):
    rows = {}
    for line in Path(path).open():
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        tile = row["tile"]
        key = (row["recording"], Path(row["source"]["shard"]).name,
            tile.get("ownership_start_timebin", tile["start_timebin"]),
            tile.get("ownership_end_timebin", tile["end_timebin"]))
        if "events" in row:
            retry = row.get("adjudicated") and not row["events"] and any(x.get("events") for x in row.get("passes", [])[:-1])
            if retry:
                rows.pop(key, None)
                continue
            row = {**row, "boxes": [event for event in row["events"] if event["label"] in FOREGROUND]}
        rows[key] = row
    rows = list(rows.values())
    random.Random(seed).shuffle(rows)
    return rows[:maximum] if maximum else rows


def split_rows(rows, fraction, seed):
    recordings = sorted({row["recording"] for row in rows})
    random.Random(seed).shuffle(recordings)
    count = min(len(recordings) - 1, max(1, round(len(recordings) * fraction)))
    validation = set(recordings[:count])
    return ([row for row in rows if row["recording"] not in validation],
        [row for row in rows if row["recording"] in validation])


class PixelWindows(Dataset):
    def __init__(self, rows, shard_dir, config):
        self.shard_dir, self.config, self.windows = Path(shard_dir), config, []
        width, mels = config.num_timebins, config.mels
        for row in rows:
            tile = row["tile"]
            for start in range(tile["start_timebin"], tile["end_timebin"], width):
                valid = min(width, tile["end_timebin"] - start)
                target = np.zeros((mels, width), np.float32)
                for box in row["boxes"]:
                    left, right = max(start, box["start_timebin"]), min(start + valid, box["end_timebin"])
                    if left < right:
                        target[box["low_mel_bin"]:box["high_mel_bin"], left - start:right - start] = 1
                self.windows.append((row, start, valid, target))

    def __getitem__(self, index):
        row, start, valid, target = self.windows[index]
        source = row["source"]
        raw = np.zeros((self.config.mels, self.config.num_timebins), np.float32)
        shard = self.shard_dir / Path(source["shard"]).name
        raw[:, :valid] = load_spec_slice(shard, source["start"] + start, source["start"] + start + valid)
        spec = (raw - self.config.audio_mean) / self.config.audio_std
        return torch.from_numpy(spec).unsqueeze(0), torch.from_numpy(target), valid

    def __len__(self):
        return len(self.windows)
