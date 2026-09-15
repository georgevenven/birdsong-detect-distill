#!/usr/bin/env python3
"""Same five excerpts, with evaluated hard-BCE Large masks and no extra contour cleanup."""
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from birdsong_detect_distill.benchmark_data import digest, recordings, write_json
from evaluate_2d import references
from evaluate_songmae_smoothing import smooth, truth_and_coverage


def main():
    root = Path("results/qwen_teacher_powdermill")
    previous_path = root / "postprocessing_steps_2026-09-08/examples.json"
    previous = json.loads(previous_path.read_text())
    comparison_path = root / "training_loss_ablation_2026-09-08/comparison.json"
    comparison = json.loads(comparison_path.read_text())
    out = root / "hard_bce_figures_2026-09-08/qualitative"
    if out.exists():
        raise ValueError("qualitative figures already exist")
    checkpoint = comparison["checkpoints"]["hard_no_tv"]
    if digest(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
        raise ValueError("hard-BCE checkpoint changed")
    coverage = json.loads(Path("results/baselines_matched_2026-09-07/manifest.json").read_text())["powdermill"]["evaluation"]
    inventory = {r["name"]: r for r in recordings(Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw"), "powdermill")}
    cache = Path("artifacts/training_loss_ablation_2026-09-08")
    expected = {r["name"]: r for r in comparison["per_recording"]["hard_no_tv"]}
    report = dict(comparison=str(comparison_path), comparison_sha256=digest(comparison_path),
        checkpoint=checkpoint["checkpoint"], checkpoint_sha256=checkpoint["checkpoint_sha256"],
        prior_examples_sha256=digest(previous_path), selection="same five excerpts; no reselection by new performance",
        smoothing="probability-space Gaussian sigma=(2 mel bins,3 frames); full segment before crop",
        threshold=.08, mask_cleanup="none", figure_inches=[3.5, 3.5], png_dpi=600, examples=[])
    (out / "arrays").mkdir(parents=True)
    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    for name in sorted({f"Recording_1_Segment_{x['segment']:02d}" for x in previous["examples"]}):
        paths = list(cache.glob(f"predictions*/{name}.npz"))
        if len(paths) != 1:
            raise ValueError("missing or ambiguous ablation prediction cache")
        path = paths[0]
        source = json.loads(path.with_suffix(".json").read_text())["source"]
        if digest(path) != source["prediction_sha256"]:
            raise ValueError("prediction cache changed")
        with np.load(path) as data:
            probability = smooth(data["hard_no_tv"])
        mask = probability.astype(np.float64) >= .08
        width = int(np.ceil(source["duration"] * 200))
        truth, kept = truth_and_coverage(inventory[name], width, coverage[name]["intervals_seconds"])
        p, t = mask[:, :width][:, kept], truth[:, kept]
        counts = [int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())]
        if counts != expected[name]["area"]["full"]["counts"][8]:
            raise ValueError("figure mask differs from the evaluated mask")
        for example in previous["examples"]:
            if name != f"Recording_1_Segment_{example['segment']:02d}":
                continue
            start = example["start_seconds"]
            if not any(a <= start and start + 5 <= b for a, b in coverage[name]["intervals_seconds"]):
                raise ValueError("excerpt is outside matched reporting coverage")
            old = previous_path.parent / example["cache"]
            if digest(old) != example["cache_sha256"]:
                raise ValueError("reference excerpt changed")
            span = slice(start * 200, (start + 5) * 200)
            with np.load(old) as data:
                arrays = dict(raw=data["raw"], events=data["events"], probability=probability[:, span], mask=mask[:, span])
            stem = f"{example['id']}_R1S{example['segment']:02d}_{start}-{start+5}s_hard_bce"
            array_path = out / "arrays" / f"{stem}.npz"
            np.savez_compressed(array_path, **arrays)
            fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True, sharey=True)
            fig.subplots_adjust(left=.17, right=.98, bottom=.14, top=.92, hspace=.42)
            for axis in axes:
                axis.imshow(arrays["raw"], origin="lower", aspect="auto", extent=(0, 5, 0, 128),
                    cmap="magma", vmin=-70, vmax=0, interpolation="nearest", rasterized=True)
                axis.set(xlim=(0, 5), ylim=(0, 128))
            for left, low, right, high in references(arrays["events"]):
                axes[0].add_patch(Rectangle((left, low), right-left, high-low, fill=False, edgecolor="#00D5FF", linewidth=.6))
            if arrays["mask"].any():
                axes[1].contour((np.arange(1002)-.5)/200, np.arange(130)-.5, np.pad(arrays["mask"], 1),
                    levels=[.5], colors="#00D5FF", linewidths=.55)
            positions = (librosa.hz_to_mel([1000, 2000, 4000, 8000, 16000])-librosa.hz_to_mel(20))*128/(
                librosa.hz_to_mel(16000)-librosa.hz_to_mel(20))
            axes[0].set_yticks(positions, labels=["1", "2", "4", "8", "16"])
            axes[0].set_title("(a) Powdermill ground truth")
            axes[1].set_title("(b) SongMAE-Large (10k s)")
            axes[1].set_xticks(range(6))
            axes[1].set_xlabel("Time (s)")
            fig.text(.025, .53, "Frequency (kHz; mel-spaced)", rotation=90, va="center", ha="center", fontsize=9)
            for suffix in (".png", ".pdf", ".svg"):
                fig.savefig(out / (stem + suffix), dpi=600, facecolor="white")
            plt.close(fig)
            report["examples"].append(dict(id=example["id"], recording=name, start_seconds=start,
                file=stem, arrays=str(array_path.relative_to(out)), arrays_sha256=digest(array_path),
                source_prediction_sha256=source["prediction_sha256"], full_segment_mask_counts_verified=True))
            print(stem, flush=True)
    report["examples"].sort(key=lambda r: r["id"])
    report["plotter_sha256"] = digest(__file__)
    write_json(out / "examples.json", report)


if __name__ == "__main__":
    main()
