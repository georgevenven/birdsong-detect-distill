#!/usr/bin/env python3
"""Four training losses; unchanged SongMAE inference and Powdermill reporting coverage."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, area, summarize
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import load_heads, songmae_probabilities
from evaluate_songmae_smoothing import load_prediction, smooth, truth_and_coverage


RUN = "training_loss_ablation_2026-09-08"
ARTIFACTS = Path("artifacts") / RUN
RESULTS = Path("results/qwen_teacher_powdermill") / RUN
MANIFEST = Path("results/baselines_matched_2026-09-07/manifest.json")
PREVIOUS = Path("results/qwen_teacher_powdermill/probability_smoothing_2026-09-08/comparison.json")
CONDITIONS = {"soft_tv": (True, .001), "soft_no_tv": (True, 0.),
    "hard_tv": (False, .001), "hard_no_tv": (False, 0.)}


def checkpoint_metadata(previous):
    metadata = {}
    for name, (soft, tv) in CONDITIONS.items():
        path = ARTIFACTS / f"{name}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected = dict(backbone_id=previous["inference"]["backbone"],
            backbone_revision=previous["inference"]["backbone_revision"], seed=0,
            target_smoothing=soft, tv_weight=tv,
            annotations_sha256="baed2ace91476d090a1155b2fca902a07521f42f61e881b8cab82755890446dd")
        if any(saved[key] != value for key, value in expected.items()):
            raise ValueError(f"unexpected training configuration: {name}")
        if saved["metrics"]["train_windows"] != 2103 or saved["metrics"]["epochs"] != 3:
            raise ValueError("training coverage or epoch count changed")
        config = saved["training_config"]
        if any(config[key] != value for key, value in dict(optimizer="AdamW", learning_rate=.001,
                weight_decay=.0001, batch_size=16, accumulation=1, train_all=True).items()):
            raise ValueError("optimizer or checkpoint-selection protocol changed")
        metadata[name] = {key: value for key, value in saved.items() if key != "head"}
        metadata[name].update(checkpoint=str(path), checkpoint_sha256=digest(path))
    control = metadata["soft_tv"]
    for name, saved in metadata.items():
        if saved["initial_head_sha256"] != control["initial_head_sha256"]:
            raise ValueError(f"different initialization: {name}")
        if saved["training_recording_ids"] != control["training_recording_ids"]:
            raise ValueError(f"different training recordings: {name}")
        for key in ("sample_order_sha256", "cpu_rng_sha256"):
            if [r[key] for r in saved["training_history"]] != [r[key] for r in control["training_history"]]:
                raise ValueError(f"different training order/RNG: {name}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw"))
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    out = RESULTS / f"part{args.shard}.json"
    cache = ARTIFACTS / f"predictions{args.shard}"
    if out.exists() or cache.exists():
        raise ValueError("output already exists; preserve it before rerunning")
    previous = json.loads(PREVIOUS.read_text())
    if digest(MANIFEST) != previous["manifest_sha256"]:
        raise ValueError("reporting coverage changed")
    if previous["threshold_indices"]["gaussian_probability"] != {"full": 8}:
        raise ValueError("expected frozen threshold 0.08")
    if previous["smoothing"]["sigma_mel_time"] != [2, 3]:
        raise ValueError("smoothing parameters changed")
    for path, expected in previous["inference"]["code_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"native inference code changed: {path}")
    metadata = checkpoint_metadata(previous)
    legacy = Path(previous["inference"]["checkpoint"])
    if digest(legacy) != previous["inference"]["checkpoint_sha256"]:
        raise ValueError("previous control checkpoint changed")
    paths = [ARTIFACTS / f"{name}.pt" for name in CONDITIONS] + [legacy]
    device = torch.device("cuda:0")
    backbone = load_backbone(previous["inference"]["backbone"], device,
        revision=previous["inference"]["backbone_revision"])
    heads = load_heads(paths, backbone, device)
    powdermill = json.loads(MANIFEST.read_text())["powdermill"]
    names = sorted(powdermill["evaluation"])[args.shard::2]
    inventory = {r["name"]: r for r in recordings(args.root, "powdermill")}
    audio_hash = digest(args.root / "powdermill/wav_Files.zip")
    legacy_cache = Path("artifacts/baselines_matched_2026-09-07/paper/songmae")
    cache.mkdir(parents=True)
    scored = {name: [] for name in CONDITIONS}
    sources = {}
    for index, name in enumerate(names, 1):
        record = inventory[name]
        waveform, rate = read_audio(record)
        duration = len(waveform) / rate
        audio, _ = model_audio(waveform, rate, "songmae")
        predictions = songmae_probabilities(backbone, heads, audio, device)
        cached, old_source, width = load_prediction(legacy_cache, record, audio_hash)
        if duration != old_source["duration"] or not np.array_equal(predictions.pop(legacy.stem), cached):
            raise ValueError(f"native inference no longer reproduces the previous control cache: {name}")
        del cached
        intervals = powdermill["evaluation"][name]["intervals_seconds"]
        truth, kept = truth_and_coverage(record, width, intervals)
        path = cache / f"{name}.npz"
        np.savez_compressed(path, **predictions)
        source = dict(audio_sha256=audio_hash, member=record["member"], duration=duration,
            reference_sha256=hashlib.sha256(record["events"].tobytes()).hexdigest(),
            prediction_sha256=digest(path), legacy_prediction_reproduction="bitwise exact")
        sources[name] = source
        for condition, probability in predictions.items():
            # Smooth the full native grid (including its STFT endpoint) before any scoring crop.
            score = area(smooth(probability)[:, :width][:, kept], truth[:, kept])
            scored[condition].append(dict(name=name, group=record["group"],
                seconds=float(kept.sum() / RATE), area={"full": score}))
        write_json(path.with_suffix(".json"), dict(source=source,
            per_condition={condition: rows[-1] for condition, rows in scored.items()}))
        print(f"shard {args.shard}: {index}/{len(names)} {name}; legacy cache bitwise exact", flush=True)
    report = dict(shard=args.shard, shards=2, role="Powdermill development ablation; single seed",
        previous=str(PREVIOUS), previous_sha256=digest(PREVIOUS), manifest_sha256=digest(MANIFEST),
        checkpoints=metadata, smoothing=previous["smoothing"], threshold=.08,
        threshold_selection="none: retain previous control's calibration on original Recordings 2–4",
        binary_postprocessing=previous["binary_postprocessing"], aggregation=previous["aggregation"],
        area_ap_method=previous["area_ap_method"], empty_union_iou=1,
        evaluated_seconds=sum(row["seconds"] for row in scored["soft_tv"]),
        summary={name: summarize(rows, {"full": 8})["full"] for name, rows in scored.items()},
        per_recording=scored, sources=sources,
        code_sha256={path: digest(path) for path in ("scripts/train.py", "scripts/run_training_loss_ablation.sh",
            "scripts/evaluate_training_loss_ablation.py", "scripts/evaluate_songmae_smoothing.py",
            "src/birdsong_detect_distill/data.py", "src/birdsong_detect_distill/benchmark_metrics.py",
            *previous["inference"]["code_sha256"])},
        versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "librosa", "transformers")})
    write_json(out, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
