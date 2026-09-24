#!/usr/bin/env python3
"""Detect bird vocalizations in audio files with a trained SongMAE detector.

Uses the paper's inference: 128-mel spectrogram (32 kHz, 5 ms hop, 20-16000 Hz), 5-s windows
with 2.5-s overlap, Gaussian smoothing sigma=(2 mel bins, 3 frames), one probability threshold.
Writes <name>.npz (probability, mask) and <name>.csv (one row per connected detected region).
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy import ndimage

from birdsong_detect_distill.full_encoder_linear import load_heads
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import songmae_probabilities
from evaluate_songmae_smoothing import smooth

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("audio", type=Path, nargs="+")
parser.add_argument("--checkpoint", type=Path, required=True, help="trained detector (model.pt)")
parser.add_argument("--backbone", help="local copy of the pretrained backbone (default: Hugging Face id in checkpoint)")
parser.add_argument("--threshold", type=float, default=0.04, help="paper's Powdermill-calibrated value for SongMAE-Large")
parser.add_argument("--out", type=Path, default=Path("predictions"))
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

device = torch.device(args.device)
saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
backbone = load_backbone(args.backbone or saved["backbone_id"], device, None if args.backbone else saved["backbone_revision"])
if args.backbone:  # a local snapshot carries no Hugging Face revision; it must be the checkpoint's
    backbone.config._commit_hash = saved["backbone_revision"]
head = load_heads([args.checkpoint], backbone, device)[args.checkpoint.stem]
hertz = librosa.mel_frequencies(n_mels=128 + 2, fmin=20, fmax=16000)  # mel-bin edges
args.out.mkdir(parents=True, exist_ok=True)

for path in args.audio:
    waveform, _ = librosa.load(path, sr=32000, mono=True)
    with torch.inference_mode():
        probability = smooth(songmae_probabilities(backbone, {"model": head}, waveform, device)["model"])
    mask = probability >= args.threshold
    np.savez_compressed(args.out / f"{path.stem}.npz", probability=probability.astype(np.float16), mask=mask)
    labels, count = ndimage.label(mask)
    with open(args.out / f"{path.stem}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start_s", "end_s", "low_hz", "high_hz", "peak_probability"])
        regions = sorted(enumerate(ndimage.find_objects(labels), 1), key=lambda item: item[1][1].start)
        for index, (rows, cols) in regions:
            peak = float(probability[rows, cols][labels[rows, cols] == index].max())
            writer.writerow([f"{cols.start * 0.005:.3f}", f"{cols.stop * 0.005:.3f}",
                f"{hertz[rows.start]:.0f}", f"{hertz[rows.stop + 1]:.0f}", f"{peak:.3f}"])
    print(f"{path.name}: {count} regions, {mask.mean():.1%} of time-frequency bins")
