#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = {
    "direct": ("Direct", "#666666"),
    "reasoning": ("+ Reasoning", "#0072B2"),
    "self_review": ("+ Self-review", "#009E73"),
    "shifted_review": ("+ Shifted review", "#CC79A7"),
    "full": ("+ Adjudication", "#D55E00"),
}


def main():
    parser = argparse.ArgumentParser(description="Plot Qwen pipeline ablations against labeled audio seconds.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = json.loads(args.results.read_text())["models"]
    values = []
    for name, result in models.items():
        match = re.fullmatch(r"(direct|reasoning|self_review|shifted_review|full)-s(\d+)", name)
        if match:
            values.append((*match.groups(), result))
    seconds = sorted({int(value[1]) for value in values})
    if any(sum(name == variant for name, _, _ in values) != len(seconds) for variant in VARIANTS):
        raise ValueError("each variant must have the same time budgets")

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(5.6, 4.4), dpi=200)
    scores = []
    for variant, (label, color) in VARIANTS.items():
        points = sorted((int(seconds), result["iou_2d"]) for name, seconds, result in values if name == variant)
        x, y = zip(*points)
        scores.extend(y)
        axis.plot(x, y, marker="o", markersize=6, linewidth=2, color=color, label=label)
    margin = max(.01, (max(scores) - min(scores)) * .15)
    axis.set_xscale("log")
    axis.set_xticks(seconds, labels=[f"{value:,} s" for value in seconds])
    axis.set_ylim(max(0, min(scores) - margin), min(1, max(scores) + margin))
    axis.set_xlabel("Qwen-labeled training audio")
    axis.set_ylabel("Mean 2D IoU ↑")
    axis.set_title("Time–frequency localization")
    axis.grid(alpha=.18)
    axis.legend(frameon=False, fontsize=8)
    axis.set_box_aspect(1)
    fig.suptitle("Qwen annotation pipeline ablation — preliminary", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 300), (".pdf", None), (".svg", None)):
        fig.savefig(args.out.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
