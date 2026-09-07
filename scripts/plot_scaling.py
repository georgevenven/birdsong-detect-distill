#!/usr/bin/env python3
"""Plot Powdermill mask scores against SongMAE size and Qwen recording count."""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"

# Visual language follows SongMAE's plot_beans_parameters.py at origin/main 1cd3590.

MODELS = {
    "micro": ("Micro (1.75M)", "#56B4E9", 1.7, .82),
    "base": ("Base (14.89M)", "#0072B2", 2.0, .9),
    "large": ("Large (98.65M)", "#D55E00", 2.4, 1.0),
}


def rows(path):
    models = json.loads(path.read_text())["models"]
    output = []
    for name, result in models.items():
        match = re.fullmatch(r"songmae-(micro|base|large)-32x1-qwen-r(\d+)", name)
        if match:
            size, recordings = match.groups()
            output.append((size, int(recordings), result))
    assert len(output) == 12, f"expected 12 runs, found {len(output)}"
    return output


def panel(axis, values, key, title, ylabel):
    scores = []
    for size, (label, color, width, alpha) in MODELS.items():
        points = sorted((count, score[key]) for model, count, score in values if model == size)
        x, y = zip(*points)
        scores.extend(y)
        axis.plot(x, y, color=color, marker="o", markersize=6, linewidth=width,
            alpha=alpha, label=label, zorder=4)
    margin = max(.01, (max(scores) - min(scores)) * .14)
    axis.set_xlim(0, 320)
    axis.set_ylim(max(0, min(scores) - margin), min(1, max(scores) + margin))
    axis.set_title(title, fontsize=13)
    axis.set_xlabel("Qwen recordings")
    axis.set_ylabel(ylabel)
    axis.set_box_aspect(1)
    axis.set_xticks([10, 50, 100, 300], labels=["10", "50", "100", "300"])
    axis.grid(alpha=.18)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    values = rows(args.results)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), dpi=200)
    panel(axes[0], values, "temporal_iou", "Temporal localization", "Mean temporal IoU ↑")
    panel(axes[1], values, "iou_2d", "Time–frequency localization", "Mean 2D IoU ↑")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.subplots_adjust(left=.08, right=.995, bottom=.15, top=.91, wspace=.22)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "songmae_powdermill_scaling.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_name(f"{output.stem}_hq.png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
