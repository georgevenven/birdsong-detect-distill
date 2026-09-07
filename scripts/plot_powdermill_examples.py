#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from birdsong_detect_distill.evaluation import powdermill
from birdsong_detect_distill.model import clean_mask, load_detector
from evaluate_2d import reference_mask, references, songmae_probability
from evaluate_qwen_teacher import score_masks


EXAMPLES = (("A", "Sparse vocalizations", 13, 105), ("B", "Repeated song phrase", 8, 80),
    ("C", "Overlapping frequency bands", 27, 225), ("D", "Dense mixture", 7, 35))
POSTPROCESS = {"low": .22, "high": .45, "sigma": (2, 3), "minimum_area": 32, "minimum_width": 3}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_example(raw, mask, events, path, postprocessed=False):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(3.5, 3.5))
    fig.subplots_adjust(left=.17, right=.98, bottom=.14, top=.92, hspace=.52 if postprocessed else .40)
    extent = (0, 5, 0, 128)
    color = "#00D5FF"
    for axis in axes:
        axis.imshow(raw, origin="lower", aspect="auto", extent=extent, cmap="magma",
            vmin=-70, vmax=0, interpolation="nearest", rasterized=True)
    for left, low, right, high in references(events):
        axes[0].add_patch(Rectangle((left, low), right - left, high - low,
            fill=False, edgecolor=color, linewidth=.8))
    if mask.any():
        axes[1].contour((np.arange(mask.shape[1] + 2) - .5) / 200, np.arange(130) - .5,
            np.pad(mask, 1), levels=[.5], colors=color, linewidths=.7)
    frequencies = np.array([1000, 2000, 4000, 8000, 16000])
    positions = (librosa.hz_to_mel(frequencies) - librosa.hz_to_mel(20)) * 128 / (
        librosa.hz_to_mel(16000) - librosa.hz_to_mel(20))
    axes[0].set_yticks(positions, labels=["1", "2", "4", "8", "16"])
    axes[0].set(xlim=(0, 5), ylim=(0, 128), title="(a) Powdermill ground truth")
    title = "(b) SongMAE-Large (10k s)" + ("\nPost-processed" if postprocessed else "")
    axes[1].set(title=title, xlabel="Time (s)")
    axes[1].set_xticks(range(6))
    fig.text(.035, .53, "Frequency (kHz)", rotation=90, ha="center", va="center", fontsize=10)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(suffix), dpi=600, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Paper-ready Powdermill ground-truth and 10k SongMAE comparisons.")
    parser.add_argument("--root", type=Path, default=Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw"))
    parser.add_argument("--comparison", type=Path, default=Path(
        "results/qwen_teacher_powdermill/backbone_self_review_10000s_2026-09-07/large/comparison.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--postprocess", action="store_true", help="Use the standard clean_mask pipeline on full segments before cropping.")
    args = parser.parse_args()
    name = "powdermill_examples_" + ("postprocessed_" if args.postprocess else "") + "2026-09-07"
    args.out = args.out or Path("results/qwen_teacher_powdermill") / name
    args.cache = args.cache or Path("artifacts") / name
    args.out.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    if args.render_only:
        report = json.loads((args.out / "examples.json").read_text())
    else:
        if (args.out / "examples.json").exists():
            raise ValueError("examples already exist; use --render-only or a new output directory")
        comparison = json.loads(args.comparison.read_text())
        checkpoint = Path(comparison["checkpoint"])
        if digest(checkpoint) != comparison["checkpoint_sha256"]:
            raise ValueError("checkpoint differs from the evaluated model")
        for path, expected in comparison["code_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"evaluation code changed: {path}")
        device = torch.device(args.device)
        backbone, head, saved = load_detector(checkpoint, device)
        if saved["backbone_id"] != "georgeven/songmae-large-32x1" or backbone.config._commit_hash != comparison["backbone_revision"]:
            raise ValueError("unexpected backbone or revision")
        threshold = comparison["student_probability_threshold"]
        report = {"comparison": str(args.comparison), "comparison_sha256": digest(args.comparison),
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "backbone_revision": comparison["backbone_revision"], "threshold": threshold,
            "inference": {**comparison["student_inference"], "postprocessing": "clean_mask" if args.postprocess else "none"},
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "selection": "Four qualitative candidates chosen for human-annotation structure and visual variety; not a representative sample.",
            "display": {"seconds": 5, "frequency_scale": "mel", "frequency_hz": [20, 16000],
                "spectrogram_db": [-70, 0], "reference": "full_segment_maximum",
                "postprocessing": "clean_mask" if args.postprocess else "none"},
            "examples": []}
        if args.postprocess:
            report["raw_threshold"] = report.pop("threshold")
            report["postprocessing"] = {**POSTPROCESS, "scope": "full_segment_before_crop",
                "smoothing_space": "log_odds", "sigma_units": ["mel_bins", "5ms_frames"],
                "hysteresis_connectivity": 8, "closing_mel_time": [3, 5], "component_connectivity": 4,
                "fill_holes": True, "implementation_sha256": digest("src/birdsong_detect_distill/model.py")}
        args.cache.mkdir(parents=True, exist_ok=True)
        selected = {f"Recording_1_Segment_{segment:02d}" for _, _, segment, _ in EXAMPLES}
        with zipfile.ZipFile(args.root / "powdermill/annotation_Files.zip") as archive:
            tables = {Path(name).name.split(".Table")[0]: pd.read_csv(archive.open(name), sep="\t")
                for name in archive.namelist() if name.endswith(".txt") and Path(name).name.split(".Table")[0] in selected}
        for recording, waveform, events in powdermill(args.root, selected):
            probability = songmae_probability(backbone, head, waveform, device)
            if args.postprocess:
                smooth, detected = clean_mask(probability, **POSTPROCESS)
            raw = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=1024,
                hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000), ref=np.max, top_db=None)
            for identifier, description, segment, start in EXAMPLES:
                if recording != f"Recording_1_Segment_{segment:02d}":
                    continue
                if not any(a <= start and start + 5 <= b for a, b in comparison["coverage"][recording]["intervals_seconds"]):
                    raise ValueError(f"example outside scored intervals: {recording}, {start}")
                selected_events = (events[:, 0] < start + 5) & (events[:, 1] > start)
                cropped = events[selected_events].copy()
                cropped[:, 0] = np.maximum(cropped[:, 0], start) - start
                cropped[:, 1] = np.minimum(cropped[:, 1], start + 5) - start
                first, last = start * 200, (start + 5) * 200
                prob, spec = probability[:, first:last], raw[:, first:last]
                truth = reference_mask(cropped, prob.shape)
                filename = f"{identifier}_R1S{segment:02d}_{start}-{start + 5}s"
                cache = args.cache / f"{filename}.npz"
                processed = {"smoothed_probability": smooth[:, first:last], "mask": detected[:, first:last]} if args.postprocess else {}
                np.savez_compressed(cache, raw=spec, probability=prob, events=cropped, **processed)
                report["examples"].append({"id": identifier, "description": description, "recording": recording,
                    "start_seconds": start, "end_seconds": start + 5, "reference_events": len(cropped),
                    "species_codes": sorted(tables[recording].loc[selected_events, "Species"].astype(str).unique()),
                    "file": filename, "cache": str(cache), "cache_sha256": digest(cache),
                    "clip_metrics": score_masks(processed.get("smoothed_probability", prob),
                        processed.get("mask", prob >= threshold), truth)})
                print(f"prepared {identifier}: {recording}, {start}–{start + 5} s", flush=True)
        del backbone, head
        torch.cuda.empty_cache()
        report["examples"].sort(key=lambda row: row["id"])
    for row in report["examples"]:
        if digest(row["cache"]) != row["cache_sha256"]:
            raise ValueError(f"changed cached example: {row['id']}")
        with np.load(row["cache"]) as data:
            postprocessed = report["display"]["postprocessing"] == "clean_mask"
            mask = data["mask"] if postprocessed else data["probability"] >= report["threshold"]
            draw_example(data["raw"], mask, data["events"], args.out / row["file"], postprocessed)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5), layout="constrained")
    for axis, row in zip(axes.flat, report["examples"]):
        axis.imshow(plt.imread(args.out / f"{row['file']}.png"))
        axis.set_title(f"{row['id']}: {row['description']}", fontsize=11)
        axis.set_axis_off()
    fig.savefig(args.out / "00_contact_sheet.png", dpi=300, facecolor="white")
    plt.close(fig)
    report["plotter_sha256"] = digest(__file__)
    (args.out / "examples.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.out / "00_contact_sheet.png")


if __name__ == "__main__":
    main()
