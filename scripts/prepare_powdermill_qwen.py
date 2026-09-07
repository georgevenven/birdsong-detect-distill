#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import librosa
import numpy as np

from birdsong_detect_distill.evaluation import powdermill


def main():
    parser = argparse.ArgumentParser(description="Prepare Powdermill spectrograms for direct Qwen annotation.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/powdermill/qwen"))
    args = parser.parse_args()
    shards = args.out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (recording, waveform, _) in enumerate(powdermill(args.root)):
        spec = librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=1024,
            hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000)
        spec = librosa.power_to_db(spec, ref=np.max, top_db=None).astype(np.float32)
        shard = f"powdermill_{index:03d}.npy"
        np.save(shards / shard, spec.T)
        rows.append({"status": "ok", "recording": recording,
            "source": {"shard": shard, "start": 0, "end": spec.shape[1]},
            "tile": {"start_timebin": 0, "end_timebin": spec.shape[1]}})
        print(f"{index + 1}: {recording}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audio_params.json").write_text(json.dumps(
        {"sr": 32000, "hop_size": 160, "mels": 128}, indent=2) + "\n")
    (args.out / "recordings.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    print(json.dumps({"recordings": len(rows), "timebins": sum(row["source"]["end"] for row in rows),
        "five_second_windows": sum((row["source"]["end"] + 999) // 1000 for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
