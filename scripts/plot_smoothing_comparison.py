#!/usr/bin/env python3
"""Draw the same five Powdermill excerpts using the masks from the matched smoothing experiment."""
import argparse
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from birdsong_detect_distill.benchmark_data import digest, recordings, write_json
from evaluate_2d import references
from evaluate_songmae_smoothing import CONDITIONS, REPO, SIGMA, load_prediction, smooth, truth_and_coverage


def draw(data, example, report, out):
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.97, bottom=.15, top=.84, hspace=.48)
    fig.suptitle(f"{example['id']}: {example['description']}\nSongMAE-Large · 10,000 s self-review labels", fontsize=13, y=.97)
    color, extent = "#00D5FF", (0, 5, 0, 128)
    for axis in axes:
        axis.imshow(data["raw"], origin="lower", aspect="auto", extent=extent, cmap="magma",
            vmin=-70, vmax=0, interpolation="nearest", rasterized=True)
        axis.set(xlim=(0, 5), ylim=(0, 128))
    for left, low, right, high in references(data["events"]):
        axes[0].add_patch(Rectangle((left, low), right - left, high - low,
            fill=False, edgecolor=color, linewidth=.7))
    axes[0].set_title("(a) Powdermill ground truth", fontsize=11, pad=6)
    for axis, condition, prefix in zip(axes[1:], CONDITIONS, ("(b) Raw probabilities", "(c) Gaussian-smoothed probabilities")):
        mask = data[condition + "_mask"]
        if mask.any():
            axis.contour((np.arange(1002) - .5) / 200, np.arange(130) - .5,
                np.pad(mask, 1), levels=[.5], colors=color, linewidths=.65)
        threshold = report["summary"][condition]["threshold"]
        axis.set_title(f"{prefix} · threshold {threshold:.2f}", fontsize=11, pad=6)
    positions = (librosa.hz_to_mel([1000, 2000, 4000, 8000, 16000]) - librosa.hz_to_mel(20)) * 128 / (
        librosa.hz_to_mel(16000) - librosa.hz_to_mel(20))
    axes[0].set_yticks(positions, labels=["1", "2", "4", "8", "16"])
    axes[-1].set_xticks(range(6))
    axes[-1].set_xlabel("Time within excerpt (s)", fontsize=11)
    fig.text(.035, .5, "Frequency (kHz; mel-spaced)", rotation=90, va="center", ha="center", fontsize=11)
    fig.text(.5, .03, f"R1 segment {example['segment']:02d}, {example['start_seconds']}–{example['start_seconds'] + 5} s. "
        "σ = 2 mel bins × 3 frames.\nThresholds selected on R2–R4. Cyan contours = evaluated masks; no further cleanup.",
        ha="center", va="center", fontsize=8.5)
    for suffix in (".png", ".pdf"):
        fig.savefig((out / example["file"]).with_suffix(suffix), dpi=400, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--examples", type=Path, default=Path(
        "results/qwen_teacher_powdermill/postprocessing_steps_2026-09-08/examples.json"))
    parser.add_argument("--root", type=Path, default=Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/baselines_matched_2026-09-07/paper/songmae"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    out = args.report.parent / "figures"
    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42})
    if args.render_only:
        examples = json.loads((out / "examples.json").read_text())
        if examples["comparison_sha256"] != digest(args.report):
            raise ValueError("changed comparison report")
    else:
        if out.exists():
            raise ValueError("choose a new experiment or --render-only")
        for path, expected in report["code_sha256"].items():
            if digest(REPO / path) != expected:
                raise ValueError(f"changed evaluation code: {path}")
        if list(SIGMA) != report["smoothing"]["sigma_mel_time"]:
            raise ValueError("smoothing settings differ from evaluated values")
        metadata = json.loads((args.cache / "metadata.json").read_text())
        if metadata != report["inference"]:
            raise ValueError("cached model differs from evaluation")
        for path, expected in metadata["code_sha256"].items():
            if digest(REPO / path) != expected:
                raise ValueError(f"changed inference code: {path}")
        prior = json.loads(args.examples.read_text())
        if prior["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("previous examples use a different checkpoint")
        manifest = json.loads(Path(report["manifest"]).read_text())
        if digest(report["manifest"]) != report["manifest_sha256"]:
            raise ValueError("changed coverage manifest")
        inventory = {r["name"]: r for r in recordings(args.root, "powdermill")}
        audio_hash = digest(args.root / "powdermill/wav_Files.zip")
        out.mkdir()
        (out / "arrays").mkdir()
        examples = dict(comparison_sha256=digest(args.report), prior_examples_sha256=digest(args.examples),
            selection="Same A–E excerpts as the previous stage figures; no reselection", examples=[], verification={})
        for segment in sorted({row["segment"] for row in prior["examples"]}):
            name = f"Recording_1_Segment_{segment:02d}"
            probability, source, width = load_prediction(args.cache, inventory[name], audio_hash)
            if source != report["sources"][name]:
                raise ValueError("changed evaluation predictions")
            values = {"raw": probability, "gaussian_probability": smooth(probability)}
            intervals = manifest["powdermill"]["evaluation"][name]["intervals_seconds"]
            truth, kept = truth_and_coverage(inventory[name], width, intervals)
            masks = {}
            for condition, score in values.items():
                threshold = report["summary"][condition]["threshold"]
                masks[condition] = score.astype(np.float64) >= threshold
                prediction, target = masks[condition][:, :width][:, kept], truth[:, kept]
                counts = [int((prediction & target).sum()), int((prediction & ~target).sum()), int((~prediction & target).sum())]
                row = next(r for r in report["per_recording"][condition]["evaluation"] if r["name"] == name)
                index = report["threshold_indices"][condition]["full"]
                if counts != row["area"]["full"]["counts"][index]:
                    raise ValueError("figure masks differ from actually evaluated masks")
            examples["verification"][name] = "Both full-segment masks reproduce evaluated TP/FP/FN on the matched intervals"
            for old in prior["examples"]:
                if old["segment"] != segment:
                    continue
                start = old["start_seconds"]
                if not any(a <= start and start + 5 <= b for a, b in intervals):
                    raise ValueError("example outside matched evaluation coverage")
                previous = args.examples.parent / old["cache"]
                if digest(previous) != old["cache_sha256"]:
                    raise ValueError("changed prior example")
                span = slice(start * 200, (start + 5) * 200)
                with np.load(previous) as data:
                    if not np.array_equal(probability[:, span], data["probability"]):
                        raise ValueError("previous and current excerpt predictions differ")
                    arrays = dict(raw=data["raw"], events=data["events"],
                        raw_probability=probability[:, span], smoothed_probability=values["gaussian_probability"][:, span],
                        **{key + "_mask": value[:, span] for key, value in masks.items()})
                filename = f"{old['id']}_R1S{segment:02d}_{start}-{start + 5}s_comparison"
                path = out / "arrays" / (filename + ".npz")
                np.savez_compressed(path, **arrays)
                examples["examples"].append(dict(id=old["id"], description=old["description"], segment=segment,
                    start_seconds=start, file=filename, cache=str(path.relative_to(out)), cache_sha256=digest(path)))
        examples["examples"].sort(key=lambda row: row["id"])
        write_json(out / "examples.json", examples)
    for example in examples["examples"]:
        path = out / example["cache"]
        if digest(path) != example["cache_sha256"]:
            raise ValueError("changed figure arrays")
        with np.load(path) as data:
            draw(data, example, report, out)
        print(example["file"], flush=True)
    examples["plotter_sha256"] = digest(__file__)
    write_json(out / "examples.json", examples)


if __name__ == "__main__":
    main()
