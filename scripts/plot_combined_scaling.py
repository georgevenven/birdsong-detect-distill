#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
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
    parser.add_argument("--ylabel", default="Mask AP")
    parser.add_argument("--legend-position", choices=("lower-right", "top"), default="lower-right")
    args = parser.parse_args()
    if any(args.out.with_suffix(suffix).exists() for suffix in (".png", ".pdf", ".svg", ".json")):
        raise ValueError("figure already exists; choose a new output path")
    scaling, backbones = (json.loads(path.read_text()) for path in (args.scaling, args.backbones))
    for key in ("dataset", "evaluated_seconds", "segments", "source_recordings", "aggregation", "qwen_self_review"):
        if scaling[key] != backbones[key]:
            raise ValueError(f"incompatible panel data: {key}")
    rows = sorted(scaling["students"], key=lambda row: row["training_seconds"])
    seconds = [row["training_seconds"] for row in rows]
    if seconds != [100, 1000, 10000] or backbones["training_seconds"] != 10000:
        raise ValueError("expected 100/1k/10k scaling and a fixed 10k backbone comparison")
    large = backbones["models"]["large"]
    multi = "seeds" in scaling
    checkpoint_key = "checkpoint_sha256_by_seed" if multi else "checkpoint_sha256"
    if rows[-1][checkpoint_key] != large[checkpoint_key] or not math.isclose(
            rows[-1]["mask_ap_2d"], large["mask_ap_2d"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("the highlighted settings must use the same checkpoint and reproduce its AP")
    sizes = ("micro", "base", "large")
    label_ap = [row["mask_ap_2d"] for row in rows]
    backbone_ap = [backbones["models"][size]["mask_ap_2d"] for size in sizes]
    teacher_ap = scaling["qwen_self_review"]["mask_ap_2d"]
    label_sd = backbone_sd = None
    if multi:
        if scaling["seeds"] != [0, 1, 2] or backbones.get("seeds") != scaling["seeds"]:
            raise ValueError("expected the same three training seeds in both panels")
        for row in [*rows, *backbones["models"].values()]:
            values = row["per_seed"]
            if [v["seed"] for v in values] != scaling["seeds"] or set(row[checkpoint_key]) != {"0", "1", "2"}:
                raise ValueError("incomplete seed coverage")
            if any(row[checkpoint_key][str(v["seed"])] != v["checkpoint_sha256"] for v in values):
                raise ValueError("seed checkpoint mismatch")
            if not math.isclose(row["mask_ap_2d"], statistics.mean(v["ap"] for v in values), abs_tol=1e-12):
                raise ValueError("incorrect seed mean")
            if not math.isclose(row["mask_ap_2d_std"], statistics.stdev(v["ap"] for v in values), abs_tol=1e-12):
                raise ValueError("incorrect sample standard deviation")
        label_sd = [row["mask_ap_2d_std"] for row in rows]
        backbone_sd = [backbones["models"][size]["mask_ap_2d_std"] for size in sizes]
        if label_sd[-1] != backbone_sd[-1]:
            raise ValueError("shared Large error bars differ")

    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.17, right=.98, bottom=.17,
        top=.82 if args.legend_position == "top" else .91, wspace=.16)
    left.plot(seconds, label_ap, color=".55", marker="o", linewidth=1.6, markersize=4.5)
    if multi:
        left.errorbar(seconds, label_ap, yerr=label_sd, fmt="none", ecolor=".25", elinewidth=1,
            capsize=3, capthick=1, zorder=3)
    left.plot(seconds[-1], label_ap[-1], color="C0", marker="o", markersize=6, zorder=4)
    left.set_xscale("log")
    left.set_xticks(seconds, labels=["100", "1k", "10k"])
    left.xaxis.set_minor_locator(NullLocator())
    left.set(xlim=(65, 15500), ylim=(0, 1), ylabel=args.ylabel, xlabel="Training audio (s)", title="(a) Label budget")
    left.set_yticks([0, .2, .4, .6, .8, 1])
    error_kw = dict(yerr=backbone_sd, capsize=3, error_kw=dict(ecolor=".25", elinewidth=1, capthick=1)) if multi else {}
    bars = right.bar(range(3), backbone_ap, width=.55, color=[".65", ".65", "C0"], **error_kw)
    right.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    right.set_xticks(range(3), labels=[size.title() for size in sizes])
    right.set(xlim=(-.6, 2.6), xlabel="Backbone", title="(b) Model size")
    for axis in (left, right):
        baseline = axis.axhline(teacher_ap, color=".25", linestyle="--", linewidth=1.2, zorder=3)
    if args.legend_position == "top":
        fig.legend([baseline], ["Qwen_teacher labels"], loc="upper center", bbox_to_anchor=(.58, .98),
            frameon=False, fontsize=9.5, handlelength=2, handletextpad=.6)
    else:
        left.legend([baseline], ["Qwen_teacher\nlabels"], loc="lower right", frameon=False,
            fontsize=9.5, handlelength=1.3, handletextpad=.5)
    if multi:
        fig.text(.58, .025, "Mean ± SD (3 training seeds)", ha="center", fontsize=8.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(args.out.with_suffix(suffix), dpi=600, facecolor="white")
    plt.close(fig)
    report = {"sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.scaling, args.backbones)}, "evaluated_seconds": scaling["evaluated_seconds"],
        "source_recordings": scaling["source_recordings"], "metric": "segment_mean_mask_ap_2d",
        "label_budget_seconds": seconds, "label_budget_ap": label_ap, "backbone_sizes": sizes,
        "backbone_ap": backbone_ap, "qwen_self_review_ap": teacher_ap,
        "figure_inches": [3.5, 3.5], "png_dpi": 600,
        "legend_position": args.legend_position}
    if multi:
        report.update(metric="segment_mean_mask_ap_2d_then_seed_mean", seeds=scaling["seeds"],
            label_budget_ap_std=label_sd, backbone_ap_std=backbone_sd,
            shared_checkpoints_sha256_by_seed=large[checkpoint_key],
            error_bars="plus/minus one sample standard deviation across training seeds, ddof=1")
    else:
        report["shared_checkpoint_sha256"] = large[checkpoint_key]
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
