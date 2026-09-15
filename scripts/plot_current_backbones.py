#!/usr/bin/env python3
"""Square, single-column AP-only plot of the matched Micro/Base/Large experiment."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from evaluate_current_backbones import ANCHOR, RESULTS


def main():
    figure = RESULTS / "backbone_pixel_ap"
    if any(figure.with_suffix(suffix).exists() for suffix in (".png", ".pdf", ".svg")):
        raise ValueError("figure exists; preserve it")
    anchor = json.loads(ANCHOR.read_text())
    sources, models = {str(ANCHOR): digest(ANCHOR)}, {}
    for size in ("micro", "base", "large"):
        if size == "large":
            rows, expected = anchor["per_recording"]["student_self_review"], anchor["summary"]["student_self_review"]
            checkpoint = anchor["protocol"]["checkpoints"]["self_review"]
        else:
            path = RESULTS / size / "comparison.json"
            run = json.loads(path.read_text())
            sources[str(path)] = digest(path)
            if run["protocol"]["anchor_sha256"] != digest(ANCHOR) or run["protocol"]["partitions"] != anchor["protocol"]["partitions"]:
                raise ValueError(f"unmatched Large anchor or intervals: {size}")
            rows, expected, checkpoint = run["per_recording"], run["summary"], run["protocol"]["checkpoint"]
        if digest(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed: {size}")
        measured = summarize(rows["evaluation"], calibrate(rows["calibration"]))["full"]
        if measured != expected or len(rows["evaluation"]) != 36 or sum(row["seconds"] for row in rows["evaluation"]) != 10800:
            raise ValueError(f"summary/coverage does not reproduce: {size}")
        models[size] = dict(**measured, checkpoint=checkpoint["checkpoint"],
            checkpoint_sha256=checkpoint["checkpoint_sha256"], selected_epoch=checkpoint["metrics"]["selected_epoch"])
    teacher_ap = anchor["summary"]["teacher_self_review"]["ap"]
    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 11,
        "ytick.labelsize": 10.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.19, right=.97, bottom=.18, top=.95)
    bars = axis.bar([name.title() for name in models], [row["ap"] for row in models.values()],
        width=.58, color=[".65", ".65", "C0"])
    axis.bar_label(bars, fmt="%.3f", padding=5, fontsize=11)
    axis.axhline(teacher_ap, color=".25", linestyle="--", linewidth=1.3, label="Qwen_teacher labels")
    axis.set(ylim=(0, 1), ylabel="Powdermill pixel AP", xlabel="SongMAE backbone")
    axis.set_yticks([0, .2, .4, .6, .8, 1])
    axis.legend(loc="lower right", frameon=False, fontsize=9.5, handlelength=1.4)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(figure.with_suffix(suffix), dpi=600, facecolor="white")
    plt.close(fig)
    write_json(RESULTS / "summary.json", dict(models=models, teacher_self_review_pixel_ap=teacher_ap,
        training_seconds=10000, validation_seconds=800, evaluated_seconds=10800, evaluation_segments=36,
        metric="exact segment-mean pixel AP over 35 positive Recording_1 segments", sources_sha256=sources,
        figure_inches=[3.5, 3.5], png_dpi=600))
    with (RESULTS / "backbones.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["Backbone", "Pixel AP", "2D IoU", "Threshold", "Selected epoch"])
        for name, row in models.items():
            writer.writerow([name.title(), row["ap"], row["iou"], row["threshold"], row["selected_epoch"]])
    caption = ("Effect of SongMAE backbone size on Powdermill pixel AP. All adapters use the same 10,000 s of XC "
        "training audio with self-reviewed Qwen labels, hard foreground targets and plain binary cross-entropy. "
        "Checkpoints are selected by minimum loss on 800 s of recording-disjoint XC validation audio across three epochs. "
        "Foreground probabilities are Gaussian-smoothed (2 mel bins, 15 ms). AP is computed from continuous scores "
        "and averaged over the 35 positive segments of complete Recording_1 (36 segments, 10,800 s total). "
        "The dashed line shows the self-reviewed Qwen teacher on identical intervals. Each model's IoU threshold "
        "is calibrated separately on Powdermill Recordings_2–4; thresholds do not affect AP. One training seed per size; "
        "no error bars or statistical significance claims.\n")
    (RESULTS / "caption.txt").write_text(caption)
    print(json.dumps(models, indent=2))
    print(figure.with_suffix(".png"))


if __name__ == "__main__":
    main()
