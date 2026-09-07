#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from birdsong_detect_distill.evaluation import powdermill
from birdsong_detect_distill.model import clean_mask, load_detector
from evaluate_2d import references, songmae_probability


def main():
    parser = argparse.ArgumentParser(description="Plot Powdermill boxes and SongMAE predictions.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/visualizations/powdermill"))
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, head, saved = load_detector(args.checkpoint, device)
    rng = random.Random(args.seed)
    chosen = set(rng.sample(range(77), args.n))
    args.out.mkdir(parents=True, exist_ok=True)

    output = 0
    for recording_index, (recording, waveform, events) in enumerate(powdermill(args.root)):
        if recording_index not in chosen:
            continue
        boxes = references(events)
        groups = {}
        for box in boxes:
            group = int((box[0] + box[2]) / 10)
            groups.setdefault(group, []).append(box)
        group = max(groups, key=lambda key: len(groups[key]))
        start, end = group * 5, (group + 1) * 5
        window = waveform[start * 32000:end * 32000]
        probability = songmae_probability(backbone, head, window, device)
        raw = librosa.power_to_db(librosa.feature.melspectrogram(y=window, sr=32000, n_fft=1024,
            hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000), ref=np.max, top_db=None)
        smooth, detected = clean_mask(probability)

        fig = plt.figure(figsize=(14, 9), layout="constrained")
        grid = fig.add_gridspec(3, 2, width_ratios=(1, .018))
        axes = [fig.add_subplot(grid[0, 0])]
        axes += [fig.add_subplot(grid[index, 0], sharex=axes[0], sharey=axes[0]) for index in (1, 2)]
        color_axis = fig.add_subplot(grid[:, 1])
        extent = (start, end, 0, 128)
        axes[0].imshow(raw, origin="lower", aspect="auto", extent=extent, cmap="magma")
        image = axes[1].imshow(probability, origin="lower", aspect="auto", extent=extent,
            cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axes[1].contour(probability, levels=[torch.sigmoid(torch.tensor(saved["threshold"])).item()],
            colors="white", linewidths=.7, origin="lower", extent=extent)
        axes[2].imshow(raw, origin="lower", aspect="auto", extent=extent, cmap="magma")
        axes[2].imshow(np.ma.masked_where(~detected, smooth), origin="lower", aspect="auto", extent=extent,
            cmap="winter", vmin=.22, vmax=1, alpha=.5, interpolation="bilinear")
        if detected.any():
            axes[2].contour(detected, levels=[.5], colors="lime", linewidths=.8, origin="lower", extent=extent)
        for left, low, right, high in groups[group]:
            axes[0].add_patch(Rectangle((max(start, left), low), min(end, right) - max(start, left), high - low,
                fill=False, edgecolor="cyan", linewidth=1.4))
        for axis, title in zip(axes, ("Human Powdermill boxes", "SongMAE Large 32×1 P(song)",
                "Cleaned SongMAE detections")):
            axis.set_title(title)
            axis.set_ylabel("Mel bin")
        axes[-1].set_xlabel("Time (s)")
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)
        fig.colorbar(image, cax=color_axis, label="P(song)")
        fig.suptitle(f"Powdermill {recording} | {start}–{end} s", fontsize=15)
        output += 1
        fig.savefig(args.out / f"powdermill_{output:02d}_{recording}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {output} Powdermill predictions to {args.out}")


if __name__ == "__main__":
    main()
