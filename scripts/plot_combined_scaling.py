#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def main():
    root = Path("results/qwen_teacher_powdermill")
    parser = argparse.ArgumentParser(description="Combine label-budget and backbone mask AP in a square figure.")
    parser.add_argument("--scaling", type=Path, default=root / "self_review_scaling_2026-09-07/scaling_single_column.json")
    parser.add_argument("--backbones", type=Path, default=root / "backbone_self_review_10000s_2026-09-07/summary.json")
    parser.add_argument("--out", type=Path, default=root / "combined_scaling_2026-09-07/xc_scaling_and_backbones")
    args = parser.parse_args()
    scaling, backbones = (json.loads(path.read_text()) for path in (args.scaling, args.backbones))
    for key in ("dataset", "evaluated_seconds", "segments", "source_recordings", "aggregation", "qwen_self_review"):
        if scaling[key] != backbones[key]:
            raise ValueError(f"incompatible panel data: {key}")
    rows = sorted(scaling["students"], key=lambda row: row["training_seconds"])
    seconds = [row["training_seconds"] for row in rows]
    if seconds != [100, 1000, 10000] or backbones["training_seconds"] != 10000:
        raise ValueError("expected 100/1k/10k scaling and a fixed 10k backbone comparison")
    large = backbones["models"]["large"]
    if rows[-1]["checkpoint_sha256"] != large["checkpoint_sha256"] or not math.isclose(
            rows[-1]["mask_ap_2d"], large["mask_ap_2d"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("the highlighted settings must use the same checkpoint and reproduce its AP")
    sizes = ("micro", "base", "large")
    label_ap = [row["mask_ap_2d"] for row in rows]
    backbone_ap = [backbones["models"][size]["mask_ap_2d"] for size in sizes]
    teacher_ap = scaling["qwen_self_review"]["mask_ap_2d"]

    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.17, right=.98, bottom=.17, top=.91, wspace=.16)
    left.plot(seconds, label_ap, color=".55", marker="o", linewidth=1.6, markersize=4.5)
    left.plot(seconds[-1], label_ap[-1], color="C0", marker="o", markersize=6, zorder=4)
    left.set_xscale("log")
    left.set_xticks(seconds, labels=["100", "1k", "10k"])
    left.xaxis.set_minor_locator(NullLocator())
    left.set(xlim=(65, 15500), ylim=(0, 1), ylabel="Mask AP", xlabel="Training audio (s)", title="(a) Label budget")
    left.set_yticks([0, .2, .4, .6, .8, 1])
    bars = right.bar(range(3), backbone_ap, width=.55, color=[".65", ".65", "C0"])
    right.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    right.set_xticks(range(3), labels=[size.title() for size in sizes])
    right.set(xlim=(-.6, 2.6), xlabel="Backbone", title="(b) Model size")
    for axis in (left, right):
        baseline = axis.axhline(teacher_ap, color=".25", linestyle="--", linewidth=1.2, zorder=3)
    left.legend([baseline], ["Qwen_teacher\nlabels"], loc="lower right", frameon=False,
        fontsize=9.5, handlelength=1.3, handletextpad=.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(args.out.with_suffix(suffix), dpi=600, facecolor="white")
    plt.close(fig)
    report = {"sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.scaling, args.backbones)}, "evaluated_seconds": scaling["evaluated_seconds"],
        "source_recordings": scaling["source_recordings"], "metric": "segment_mean_mask_ap_2d",
        "label_budget_seconds": seconds, "label_budget_ap": label_ap, "backbone_sizes": sizes,
        "backbone_ap": backbone_ap, "qwen_self_review_ap": teacher_ap,
        "shared_checkpoint_sha256": large["checkpoint_sha256"], "figure_inches": [3.5, 3.5], "png_dpi": 600}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
