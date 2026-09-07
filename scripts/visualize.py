#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from birdsong_detect_distill.data import PixelWindows, load_spec_slice, read_rows, split_rows
from birdsong_detect_distill.model import clean_mask, load_detector


def main():
    parser = argparse.ArgumentParser(description="Plot held-out dense SongMAE detections.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/songmae-large-32x1-detector.pt"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/xcl/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/visualizations"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label")
    parser.add_argument("--unique-recordings", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, head, saved = load_detector(args.checkpoint, device)
    _, rows = split_rows(read_rows(args.annotations, args.seed), .25, args.seed)
    if args.label:
        rows = [row for row in rows if any(box["label"] == args.label for box in row["boxes"])]
    random.Random(args.seed).shuffle(rows)
    if args.unique_recordings:
        rows = list({row["recording"]: row for row in rows}.values())
    args.out.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows[:args.n], 1):
        data = PixelWindows([row], args.shard_dir, backbone.config)
        spec, _, valid = data[0]
        with torch.inference_mode():
            tokens = backbone(input_values=spec[None].to(device), valid_timebins=torch.tensor([valid], device=device)).last_hidden_state
            probability = head(tokens.float(), torch.tensor([valid])).sigmoid()[0, :, :valid].cpu().numpy()
        source, tile = row["source"], row["tile"]
        start_bin = data.windows[0][1]
        raw = load_spec_slice(args.shard_dir / Path(source["shard"]).name,
            source["start"] + start_bin, source["start"] + start_bin + valid)
        start = tile["onset_ms"] / 1000
        end = start + valid * backbone.config.audio_hop_size / backbone.config.audio_sr
        smooth, detected = clean_mask(probability)
        fig = plt.figure(figsize=(14, 9), layout="constrained")
        grid = fig.add_gridspec(3, 2, width_ratios=(1, .018))
        axes = [fig.add_subplot(grid[0, 0])]
        axes += [fig.add_subplot(grid[i, 0], sharex=axes[0], sharey=axes[0]) for i in (1, 2)]
        color_axis = fig.add_subplot(grid[:, 1])
        extent = (start, end, 0, backbone.config.mels)
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
        for box in row["boxes"]:
            selected = not args.label or box["label"] == args.label
            axes[0].add_patch(Rectangle((box["onset_ms"] / 1000, box["low_mel_bin"]),
                (box["offset_ms"] - box["onset_ms"]) / 1000, box["high_mel_bin"] - box["low_mel_bin"],
                fill=False, edgecolor="cyan" if selected else "white", linestyle="-" if selected else "--",
                linewidth=1.8 if selected else .8))
            if selected:
                axes[0].text(box["onset_ms"] / 1000, box["high_mel_bin"], box["label"], color="cyan",
                    fontsize=8, va="bottom")
        for axis, title in zip(axes, ("Qwen teacher boxes", "SongMAE Large 32×1 P(song)",
                "Cleaned SongMAE detections")):
            axis.set_title(title)
            axis.set_ylabel("Mel bin")
        axes[-1].set_xlabel("Time (s)")
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)
        fig.colorbar(image, cax=color_axis, label="P(song)")
        fig.suptitle(f"Held-out recording {row['recording']}", fontsize=15)
        suffix = f"_{args.label}" if args.label else ""
        fig.savefig(args.out / f"large32x1_holdout_{index:02d}_{row['recording']}{suffix}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {min(args.n, len(rows))} held-out predictions to {args.out}")


if __name__ == "__main__":
    main()
