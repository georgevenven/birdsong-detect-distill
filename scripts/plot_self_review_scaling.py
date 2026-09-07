#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot matched Powdermill mask AP against self-reviewed training duration.")
    parser.add_argument("--result", nargs=2, action="append", metavar=("SECONDS", "JSON"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted((int(seconds), Path(path), json.loads(Path(path).read_text())) for seconds, path in args.result)
    reference = runs[0][2]
    fields = ("coverage", "coordinate_space", "ap_method", "empty_conventions", "foreground_labels",
        "teacher_results_sha256", "backbone", "backbone_revision", "code_sha256", "student_inference")
    rows = []
    for seconds, path, run in runs:
        if run["qwen_variant"] != "self_review" or any(run[key] != reference[key] for key in fields):
            raise ValueError(f"incompatible comparison: {path}")
        for key in ("selection", "segments", "source_recordings", "threshold_grid"):
            if run["calibration"][key] != reference["calibration"][key]:
                raise ValueError(f"different calibration protocol: {path}")
        rows.append({"training_seconds": seconds, "training_windows": run["training"]["train_windows"],
            "training_recordings": run["training"]["recordings"], "epochs": run["training"]["epochs"],
            "threshold": run["student_probability_threshold"], "result": str(path),
            "checkpoint": run["checkpoint"], "checkpoint_sha256": run["checkpoint_sha256"],
            **run["models"]["songmae"]})
    report = {"dataset": "powdermill", "evaluated_seconds": reference["evaluated_seconds"],
        "segments": reference["segments"], "source_recordings": reference["source_recordings"],
        "aggregation": reference["aggregation"], "qwen_self_review": reference["models"]["qwen"], "students": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    plt.style.use("default")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(3.5, 3.5), layout="constrained")
    seconds = [row["training_seconds"] for row in rows]
    values = [row["mask_ap_2d"] for row in rows]
    axis.plot(seconds, values, marker="o", color="C0", linewidth=1.6, markersize=5, label="SongMAE-Large")
    axis.axhline(reference["models"]["qwen"]["mask_ap_2d"], color="C0", linestyle="--", linewidth=1.3,
        label="Qwen self-review")
    axis.set_xscale("log")
    axis.set_xticks(seconds, labels=[f"{value:,}" for value in seconds])
    axis.set_yticks([0, .2, .4, .6, .8, 1])
    axis.set(ylim=(0, 1), ylabel="Powdermill mask AP", xlabel="Xeno-Canto training audio (s)")
    axis.margins(x=.07)
    axis.legend(loc="upper left", frameon=False)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(args.out.with_suffix(suffix), dpi=600)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
