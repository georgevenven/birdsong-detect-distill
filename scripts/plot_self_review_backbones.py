#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Summarize matched 10k-second self-review backbone comparisons.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    runs = {size: json.loads((args.results / size / "comparison.json").read_text()) for size in ("micro", "base", "large")}
    manifest = json.loads((args.results / "run.json").read_text())
    reference = runs["large"]
    expected = {"train_windows": manifest["training_windows"], "recordings": manifest["training_recordings"],
        "epochs": manifest["training"]["epochs"]}
    if reference["qwen_variant"] != "self_review" or any(reference["training"][key] != value for key, value in expected.items()):
        raise ValueError("training metadata does not match the self-review experiment")
    shared = ("coverage", "coordinate_space", "ap_method", "empty_conventions", "foreground_labels",
        "teacher_results_sha256", "qwen_variant", "training", "code_sha256", "student_inference")
    for size, run in runs.items():
        if any(run[key] != reference[key] for key in shared):
            raise ValueError(f"incompatible comparison: {size}")
        for key in ("selection", "segments", "source_recordings", "threshold_grid"):
            if run["calibration"][key] != reference["calibration"][key]:
                raise ValueError(f"different calibration protocol: {size}")
        if run["backbone"] != f"georgeven/songmae-{size}-32x1" or run["backbone_revision"] != manifest["backbone_revisions"][size]:
            raise ValueError(f"unexpected backbone: {size}")
    models = {size: {"comparison": f"{size}/comparison.json", "backbone": run["backbone"],
        "frozen_backbone_parameters": manifest["frozen_backbone_parameters"][size],
        "trainable_head_parameters": manifest["trainable_head_parameters"][size],
        "checkpoint": run["checkpoint"], "checkpoint_sha256": run["checkpoint_sha256"],
        "threshold": run["student_probability_threshold"], **run["models"]["songmae"]} for size, run in runs.items()}
    report = {"dataset": reference["dataset"], "training_seconds": manifest["training_seconds"],
        "training": reference["training"], "evaluated_seconds": reference["evaluated_seconds"],
        "source_recordings": reference["source_recordings"], "segments": reference["segments"],
        "aggregation": reference["aggregation"], "qwen_self_review": reference["models"]["qwen"], "models": models}
    (args.results / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(3.5, 3.5), layout="constrained")
    bars = axis.bar([size.title() for size in models], [row["mask_ap_2d"] for row in models.values()], width=.6)
    axis.bar_label(bars, fmt="%.3f", padding=5, fontsize=11)
    axis.axhline(report["qwen_self_review"]["mask_ap_2d"], color="C1", linestyle="--", linewidth=1.3,
        label="Qwen self-review")
    axis.set(ylim=(0, 1), xlabel="SongMAE backbone", ylabel="Powdermill mask AP")
    axis.set_yticks([0, .2, .4, .6, .8, 1])
    axis.legend(loc="upper left", frameon=False)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.results / f"backbone_mask_ap.{suffix}", dpi=600)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
