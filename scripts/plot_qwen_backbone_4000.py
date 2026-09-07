#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {
    "micro": ("Micro\n1.75M", "#56B4E9"),
    "base": ("Base\n14.89M", "#0072B2"),
    "large": ("Large\n98.65M", "#D55E00"),
}


def main():
    parser = argparse.ArgumentParser(description="Plot a fixed-data SongMAE backbone comparison.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = json.loads(args.results.read_text())["models"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4), dpi=200)
    for axis, metric, title, ylabel in zip(axes, ("temporal_iou", "iou_2d"),
            ("Temporal localization", "Time–frequency localization"),
            ("Mean temporal IoU ↑", "Mean 2D IoU ↑")):
        values = [models[size][metric] for size in MODELS]
        axis.bar(range(3), values, color=[value[1] for value in MODELS.values()], width=.68)
        axis.set_xticks(range(3), [value[0] for value in MODELS.values()])
        axis.set_ylim(max(0, min(values) - .04), min(1, max(values) + .035))
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(index, value + .004, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("SongMAE backbone size — 4,000 s adjudicated Qwen labels", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 300), (".pdf", None), (".svg", None)):
        fig.savefig(args.out.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
