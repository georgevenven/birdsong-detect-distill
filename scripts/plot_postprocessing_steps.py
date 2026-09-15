#!/usr/bin/env python3
"""Inspect each cleanup stage on cached, full-context SongMAE predictions; no GPU or aggregate evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage

from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.model import clean_mask
from evaluate_2d import references
from plot_powdermill_examples import EXAMPLES, POSTPROCESS


REPO = Path(__file__).resolve().parents[1]
MORPHOLOGY = (("hysteresis", "closing"), ("closing", "filtered"), ("filtered", "filled"))


def stages(probability):
    # Diagnostic decomposition only; checked against clean_mask on every full segment below.
    logits = np.log(np.clip(probability, 1e-5, 1 - 1e-5) / np.clip(1 - probability, 1e-5, 1))
    smooth = 1 / (1 + np.exp(-ndimage.gaussian_filter(logits, POSTPROCESS["sigma"])))
    seeds = smooth >= POSTPROCESS["high"]
    hysteresis = ndimage.binary_propagation(seeds, mask=smooth >= POSTPROCESS["low"], structure=np.ones((3, 3)))
    closing = ndimage.binary_closing(hysteresis, structure=np.ones((3, 5)))
    labels, _ = ndimage.label(closing)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= POSTPROCESS["minimum_area"]
    keep[0] = False
    for index, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds[1].stop - bounds[1].start < POSTPROCESS["minimum_width"]:
            keep[index] = False
    filtered = keep[labels]
    return dict(probability=probability, smooth=smooth, seeds=seeds, hysteresis=hysteresis,
        closing=closing, filtered=filtered, filled=ndimage.binary_fill_holes(filtered))


def crop(values, events, start):
    result = {key: value[:, start * 200:(start + 5) * 200].copy() for key, value in values.items()}
    selected = events[(events[:, 0] < start + 5) & (events[:, 1] > start)].copy()
    selected[:, 0] = np.maximum(selected[:, 0], start) - start
    selected[:, 1] = np.minimum(selected[:, 1], start + 5) - start
    return {**result, "events": selected}


def changes(before, after):
    return {"added": int((after & ~before).sum()), "removed": int((before & ~after).sum())}


def draw(data, row, out, threshold):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.subplots_adjust(left=.075, right=.97, bottom=.105, top=.885, hspace=.70, wspace=.25)
    fig.suptitle(f"{row['id']}: {row['description']}\nSongMAE-Large · 10,000 s self-review labels · "
        f"R1, segment {row['segment']:02d}, {row['start_seconds']}–{row['start_seconds'] + 5} s", fontsize=17, y=.978)
    color, extent = "#00D5FF", (0, 5, 0, 128)
    positions = (librosa.hz_to_mel([1000, 2000, 4000, 8000, 16000]) - librosa.hz_to_mel(20)) * 128 / (
        librosa.hz_to_mel(16000) - librosa.hz_to_mel(20))
    panels = [("truth", "(a) Human ground truth", "Bounding-box annotations"),
        ("probability", "(b) Raw probability", "Before post-processing"),
        ("smooth", "(c) Gaussian smoothing", "Log-odds; σ = 2 mel bins, 3 frames"),
        ("raw_mask", "(d) Raw-mask reference", f"p ≥ {threshold:.2f}; existing dev-selected threshold"),
        ("seeds", "(e) Hysteresis seeds", "Smoothed p ≥ 0.45"),
        ("hysteresis", "(f) Hysteresis growth", "Grow seeds through p ≥ 0.22; 8-connected"),
        ("closing", "(g) Morphological closing", "3 × 5 mel–time kernel"),
        ("filtered", "(h) Component filtering", "Keep ≥32 pixels and ≥3 frames; 4-connected"),
        ("filled", "(i) Hole filling / final", "Fill enclosed background holes")]
    masks = {**data, "raw_mask": data["probability"] >= threshold}
    for axis, (key, title, subtitle) in zip(axes.flat, panels):
        if key in {"probability", "smooth"}:
            artist = axis.imshow(data[key], origin="lower", aspect="auto", extent=extent,
                cmap="viridis", vmin=0, vmax=1, interpolation="nearest", rasterized=True)
            bar = fig.colorbar(artist, cax=axis.inset_axes([0, -.15, 1, .032]),
                orientation="horizontal", ticks=[0, .5, 1])
            bar.ax.tick_params(labelsize=10, pad=1)
            bar.set_label("Foreground probability", fontsize=10, labelpad=0)
        else:
            axis.imshow(data["raw"], origin="lower", aspect="auto", extent=extent, cmap="magma",
                vmin=-70, vmax=0, interpolation="nearest", rasterized=True)
            if key == "truth":
                for left, low, right, high in references(data["events"]):
                    axis.add_patch(Rectangle((left, low), right - left, high - low,
                        fill=False, edgecolor=color, linewidth=1.1))
            elif masks[key].any():
                axis.contour((np.arange(1002) - .5) / 200, np.arange(130) - .5,
                    np.pad(masks[key], 1), levels=[.5], colors=color, linewidths=.9)
        if key in row["changes"]:
            delta = row["changes"][key]
            subtitle += f"\n+{delta['added']:,} / −{delta['removed']:,} pixels vs previous step"
        axis.set_title(title + "\n" + subtitle, fontsize=11.5, pad=8)
        axis.set(xlim=(0, 5), ylim=(0, 128))
        axis.set_xticks(range(6))
        axis.set_yticks(positions, labels=["1", "2", "4", "8", "16"])
    fig.supxlabel("Time within excerpt (s)", y=.071, fontsize=15)
    fig.supylabel("Frequency (kHz; mel-spaced)", x=.019, fontsize=15)
    fig.text(.5, .033, "Cleanup path: (b) → (c) → (e) → (f) → (g) → (h) → (i). Panel (d) is a separate raw baseline.\n"
        "Cyan = foreground outlines. Full 300-s context processed before cropping. Fixed defaults; qualitative only.",
        ha="center", va="center", fontsize=11)
    for suffix in (".png", ".pdf"):
        fig.savefig((out / row["file"]).with_suffix(suffix), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/baselines_matched_2026-09-07/paper/songmae"))
    parser.add_argument("--comparison", type=Path, default=Path(
        "results/qwen_teacher_powdermill/backbone_self_review_10000s_2026-09-07/large/comparison.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    plt.style.use("default")
    plt.rcParams.update({"font.size": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42})
    if args.render_only:
        report = json.loads((args.out / "examples.json").read_text())
    else:
        if args.out.exists():
            raise ValueError("choose a new output directory, or --render-only")
        comparison = json.loads(args.comparison.read_text())
        metadata = json.loads((args.cache / "metadata.json").read_text())
        if metadata["checkpoint_sha256"] != comparison["checkpoint_sha256"] or metadata["backbone_revision"] != comparison["backbone_revision"]:
            raise ValueError("cached predictions are not from the matched 10k Large model")
        if digest(metadata["checkpoint"]) != metadata["checkpoint_sha256"]:
            raise ValueError("checkpoint changed")
        for path, expected in metadata["code_sha256"].items():
            if digest(REPO / path) != expected:
                raise ValueError(f"cached inference code differs: {path}")
        selected = {f"Recording_1_Segment_{segment:02d}" for _, _, segment, _ in EXAMPLES}
        archive_hash = digest(args.root / "powdermill/wav_Files.zip")
        pending, sources, candidate = [], {}, None
        for record in recordings(args.root, "powdermill"):
            name, events = record["name"], record["events"]
            if name not in selected:
                continue
            path = args.cache / "powdermill" / (name + ".npz")
            source = json.loads(path.with_suffix(".json").read_text())
            expected = dict(audio_sha256=archive_hash, member=record["member"],
                reference_sha256=hashlib.sha256(events.tobytes()).hexdigest())
            if source["source"] != expected or digest(path) != source["prediction_sha256"]:
                raise ValueError(f"changed source or prediction cache: {name}")
            with np.load(path) as cached:
                probability = cached["probability"]
            values = stages(probability)
            native_smooth, native_mask = clean_mask(probability, **POSTPROCESS)
            if not np.array_equal(values["smooth"], native_smooth) or not np.array_equal(values["filled"], native_mask):
                raise ValueError(f"stage decomposition differs from clean_mask: {name}")
            del native_smooth, native_mask
            waveform, rate = model_audio(*read_audio(record), "songmae")
            values["raw"] = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=rate, n_fft=1024,
                hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000), ref=np.max, top_db=None)
            segment = int(name.rsplit("_", 1)[1])
            intervals = comparison["coverage"][name]["intervals_seconds"]
            for identifier, description, number, start in EXAMPLES:
                if number == segment:
                    if not any(a <= start and start + 5 <= b for a, b in intervals):
                        raise ValueError("example outside matched coverage")
                    pending.append((identifier, description, segment, start, crop(values, events, start)))
            # Pick one diagnostic excerpt with visible morphology changes, not by reference agreement.
            changed = sum(np.count_nonzero(values[a] != values[b], axis=0) for a, b in MORPHOLOGY)
            for start in range(0, 296, 5):
                if any(number == segment and when == start for _, _, number, when in EXAMPLES):
                    continue
                if not any(a <= start and start + 5 <= b for a, b in intervals):
                    continue
                score = int(changed[start * 200:(start + 5) * 200].sum())
                if candidate is None or score > candidate[0]:
                    candidate = (score, segment, start, crop(values, events, start))
            sources[name] = dict(cache=str(path), **source, clean_mask_parity="bitwise_equal_full_segment")
            print(f"Verified full-context stages: {name}", flush=True)
        if len(pending) != 4 or candidate is None:
            raise ValueError("missing requested examples")
        score, segment, start, values = candidate
        pending.append(("E", "Morphology-change example", segment, start, values))
        args.out.mkdir(parents=True)
        (args.out / "arrays").mkdir()
        report = dict(checkpoint=metadata["checkpoint"], checkpoint_sha256=metadata["checkpoint_sha256"],
            backbone_revision=metadata["backbone_revision"], source_comparison=str(args.comparison),
            source_comparison_sha256=digest(args.comparison), inference=metadata,
            raw_threshold=comparison["student_probability_threshold"], postprocessing=POSTPROCESS,
            scope="Full segments processed before five-second crops; no GPU inference or aggregate evaluation.",
            component_connectivity=4, hysteresis_connectivity=8, closing_mel_time=[3, 5],
            selection="A–D retain the four earlier examples. E maximizes morphology-changed pixels among other "
                "nonoverlapping five-second windows in matched coverage of those four segments, without using truth accuracy.",
            fifth_example_morphology_changed_pixels=score, sources=sources, examples=[])
        for identifier, description, segment, start, values in sorted(pending):
            filename = f"{identifier}_R1S{segment:02d}_{start}-{start + 5}s_steps"
            path = args.out / "arrays" / (filename + ".npz")
            np.savez_compressed(path, **values)
            row = dict(id=identifier, description=description, segment=segment, start_seconds=start,
                file=filename, cache=str(path.relative_to(args.out)), cache_sha256=digest(path),
                changes={b: changes(values[a], values[b]) for a, b in (("seeds", "hysteresis"), *MORPHOLOGY)})
            report["examples"].append(row)
        write_json(args.out / "examples.json", report)
    for row in report["examples"]:
        path = args.out / row["cache"]
        if digest(path) != row["cache_sha256"]:
            raise ValueError("changed figure arrays")
        with np.load(path) as data:
            draw(data, row, args.out, report["raw_threshold"])
        print(f"Rendered {row['file']}", flush=True)
    report["plotter_sha256"] = digest(__file__)
    write_json(args.out / "examples.json", report)


if __name__ == "__main__":
    main()
