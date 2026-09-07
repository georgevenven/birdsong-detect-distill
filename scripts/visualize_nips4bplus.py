#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Patch, Rectangle


COLORS = {"bird": "cyan", "insect": "lime", "amphibian": "magenta", "human": "orange", "unknown": "white"}


def metadata(path):
    with path.open(newline="") as stream:
        rows = {row["class name"]: row for row in csv.DictReader(stream)}
    rows["Human"] = {"type": "human"}
    rows["Unknown"] = {"type": "unknown"}
    return rows


def annotations(path, labels):
    with path.open(newline="") as stream:
        return [(float(row[0]), float(row[0]) + float(row[1]), row[2].strip(), labels[row[2].strip()]["type"])
            for row in csv.reader(stream) if len(row) >= 3 and row[2].strip()]


def main():
    parser = argparse.ArgumentParser(description="Plot NIPS4Bplus spectrograms with strong temporal annotations.")
    parser.add_argument("--root", type=Path, default=Path("data/nips4bplus"))
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("results/visualizations/nips4bplus"))
    args = parser.parse_args()
    recordings = args.recording or ["nips4b_birds_trainfile007", "nips4b_birds_trainfile223"]
    labels = metadata(args.root / "annotations/nips4b_birdchallenge_espece_list.csv")
    args.out.mkdir(parents=True, exist_ok=True)

    for recording in recordings:
        waveform, sample_rate = sf.read(args.root / "audio" / f"{recording}.wav", dtype="float32")
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000, scale=True)
        spec = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=1024,
            hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000), ref=np.max, top_db=100)
        number = recording.removeprefix("nips4b_birds_trainfile")
        events = annotations(args.root / "annotations" / f"annotation_train{number}.csv", labels)
        duration = len(waveform) / 32000

        fig, axis = plt.subplots(figsize=(14, 5), layout="constrained")
        image = axis.imshow(spec, origin="lower", aspect="auto", extent=(0, duration, 0, 128),
            cmap="magma", vmin=-100, vmax=0)
        lanes = {kind: index for index, kind in enumerate(dict.fromkeys(event[3] for event in events))}
        for onset, offset, label, kind in events:
            color = COLORS[kind]
            axis.add_patch(Rectangle((onset, 0), offset - onset, 127.8, facecolor=color, alpha=.07,
                edgecolor=color, linewidth=1.5, linestyle="--" if kind == "unknown" else "-"))
            axis.text(onset + .015, 123 - lanes[kind] * 12, label, color=color, fontsize=8, va="top",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": .58, "pad": 1.5})

        present = list(dict.fromkeys(event[3] for event in events))
        axis.legend(handles=[Patch(facecolor="none", edgecolor=COLORS[kind], label=kind.title()) for kind in present],
            loc="lower right", framealpha=.75, ncols=len(present))
        axis.set(xlim=(0, duration), ylim=(0, 128), xlabel="Time (s)", ylabel="Mel bin",
            title=f"NIPS4Bplus {recording} | strong temporal annotations")
        fig.colorbar(image, ax=axis, label="Power (dB)", pad=.015, fraction=.025)
        fig.savefig(args.out / f"{recording}_annotated.png", dpi=150)
        plt.close(fig)
    print(f"wrote {len(recordings)} annotated spectrograms to {args.out}")


if __name__ == "__main__":
    main()
