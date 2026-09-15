#!/usr/bin/env python3
"""Refresh the AP-only scaling/backbone panels under one hard-target BCE recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, area, summarize
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import load_heads, songmae_probabilities
from evaluate_songmae_smoothing import smooth, truth_and_coverage


ARTIFACTS = Path("artifacts/hard_bce_figures_2026-09-08")
RESULTS = Path("results/qwen_teacher_powdermill/hard_bce_figures_2026-09-08")
MANIFEST = Path("results/baselines_matched_2026-09-07/manifest.json")
LOSS_RESULTS = Path("results/qwen_teacher_powdermill/training_loss_ablation_2026-09-08")
CONFIGS = ("large_s100", "large_s1000", "micro_s10000", "base_s10000")


def main():
    if (RESULTS / "comparison.json").exists():
        raise ValueError("completed figure experiment already exists")
    manifest = json.loads(MANIFEST.read_text())
    previous = json.loads((LOSS_RESULTS / "comparison.json").read_text())
    protocol = json.loads((LOSS_RESULTS / "protocol.json").read_text())
    if digest(MANIFEST) != protocol["evaluation"]["manifest_sha256"]:
        raise ValueError("coverage differs from the 10k Large ablation")
    anchor = previous["checkpoints"]["hard_no_tv"]
    if digest(anchor["checkpoint"]) != anchor["checkpoint_sha256"]:
        raise ValueError("Large 10k checkpoint changed")
    metadata = {"large_s10000": anchor}
    groups = {}
    for name in CONFIGS:
        path = ARTIFACTS / f"{name}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        size, seconds = name.split("_s")
        labels = Path(f"data/annotations/xcl/ablation_2500/ap_10000/{seconds}s/self_review.jsonl")
        if (saved["target_smoothing"] or saved["tv_weight"] != 0 or saved["seed"] != 0
                or saved["annotations_sha256"] != digest(labels) or saved["metrics"]["epochs"] != 3
                or saved["training_config"] != anchor["training_config"]
                or saved["backbone_id"] != f"georgeven/songmae-{size}-32x1"):
            raise ValueError(f"incompatible training recipe: {name}")
        metadata[name] = {k: v for k, v in saved.items() if k != "head"}
        metadata[name].update(checkpoint=str(path), checkpoint_sha256=digest(path))
        groups.setdefault((saved["backbone_id"], saved["backbone_revision"]), []).append(path)
    root = Path("/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw")
    records = {r["name"]: r for r in recordings(root, "powdermill")}
    coverage = manifest["powdermill"]["evaluation"]
    audio_hash = digest(root / "powdermill/wav_Files.zip")
    scored = {"large_s10000": previous["per_recording"]["hard_no_tv"], **{name: [] for name in CONFIGS}}
    sources = {}
    device = torch.device("cuda:0")
    for (backbone_id, revision), paths in groups.items():
        cache = ARTIFACTS / "predictions" / backbone_id.split("/")[-1]
        if cache.exists():
            raise ValueError("prediction cache exists; preserve it before rerunning")
        cache.mkdir(parents=True)
        backbone = load_backbone(backbone_id, device, revision=revision)
        heads = load_heads(paths, backbone, device)
        for index, name in enumerate(sorted(coverage), 1):
            record = records[name]
            waveform, rate = read_audio(record)
            duration = len(waveform) / rate
            audio, _ = model_audio(waveform, rate, "songmae")
            predictions = songmae_probabilities(backbone, heads, audio, device)
            width = int(np.ceil(duration * RATE))
            truth, kept = truth_and_coverage(record, width, coverage[name]["intervals_seconds"])
            path = cache / f"{name}.npz"
            np.savez_compressed(path, **predictions)
            source = dict(audio_sha256=audio_hash, member=record["member"], duration=duration,
                reference_sha256=hashlib.sha256(record["events"].tobytes()).hexdigest(), prediction_sha256=digest(path))
            sources[str(path)] = source
            for config, probability in predictions.items():
                score = area(smooth(probability)[:, :width][:, kept], truth[:, kept])
                scored[config].append(dict(name=name, group=record["group"], seconds=float(kept.sum() / RATE), area={"full": score}))
            write_json(path.with_suffix(".json"), dict(source=source,
                per_condition={config: scored[config][-1] for config in predictions}))
            print(f"{backbone_id}: {index}/{len(coverage)} {name}", flush=True)
        del backbone, heads
        torch.cuda.empty_cache()
    for rows in scored.values():
        if [r["name"] for r in rows] != sorted(coverage) or sum(r["seconds"] for r in rows) != 8235:
            raise ValueError("inconsistent reporting coverage")
    summary = {name: summarize(rows, {"full": 8})["full"] for name, rows in scored.items()}
    if any(summary["large_s10000"][key] != value for key, value in previous["summary"]["hard_no_tv"].items()
            if key != "counts_tp_fp_fn"):
        raise ValueError("shared Large 10k result changed")
    report = dict(manifest_sha256=digest(MANIFEST), training="hard masks, plain BCE, frozen backbone, final epoch 3",
        smoothing=dict(space="probability", sigma_mel_time=[2, 3], mode="reflect", truncate=4),
        threshold=.08, threshold_selection="not retuned; AP-only figure does not depend on the binary threshold",
        checkpoints=metadata, summary=summary, per_recording=scored, sources=sources,
        large_anchor=dict(path=str(LOSS_RESULTS / "comparison.json"), sha256=digest(LOSS_RESULTS / "comparison.json")),
        code_sha256={p: digest(p) for p in ("scripts/train.py", "scripts/evaluate_2d.py", "scripts/evaluate_hard_bce_figures.py",
            "scripts/evaluate_songmae_smoothing.py", "src/birdsong_detect_distill/model.py", "src/birdsong_detect_distill/data.py",
            "src/birdsong_detect_distill/benchmark_data.py", "src/birdsong_detect_distill/benchmark_metrics.py")})
    write_json(RESULTS / "comparison.json", report)
    old = json.loads(Path("results/qwen_teacher_powdermill/self_review_scaling_2026-09-07/scaling_single_column.json").read_text())
    shared = {k: old[k] for k in ("dataset", "evaluated_seconds", "segments", "source_recordings", "aggregation", "qwen_self_review")}
    def figure_row(name):
        return dict(training_seconds=int(name.split("_s")[1]), checkpoint=metadata[name]["checkpoint"],
            checkpoint_sha256=metadata[name]["checkpoint_sha256"], mask_ap_2d=summary[name]["ap"],
            iou_2d=summary[name]["iou"], training_windows=metadata[name]["metrics"]["train_windows"])
    write_json(RESULTS / "scaling.json", {**shared, "students": [figure_row(f"large_s{s}") for s in (100, 1000, 10000)]})
    write_json(RESULTS / "backbones.json", {**shared, "training_seconds": 10000,
        "models": {size: figure_row(f"{size}_s10000") for size in ("micro", "base", "large")}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
