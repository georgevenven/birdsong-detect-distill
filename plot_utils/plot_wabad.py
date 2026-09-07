#!/usr/bin/env python3
import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from birdsong_detect_distill.evaluation import powdermill, wabad
from birdsong_detect_distill.model import clean_mask, load_detector

VIEW_SECONDS = 5


def predict(waveform, backbone, head, batch_size, device):
    config = backbone.config
    power = librosa.feature.melspectrogram(y=waveform, sr=config.audio_sr, n_fft=config.audio_fft,
        hop_length=config.audio_hop_size, power=2, n_mels=config.mels, fmin=20, fmax=config.audio_sr // 2)
    raw = librosa.power_to_db(power, ref=np.max, top_db=None).astype(np.float32)
    spec = (raw - config.audio_mean) / config.audio_std
    width = config.num_timebins
    starts = list(range(0, max(1, spec.shape[1] - width + width // 2), width // 2))
    if starts[-1] + width < spec.shape[1]:
        starts.append(starts[-1] + width // 2)
    valid = [min(width, spec.shape[1] - start) for start in starts]
    windows = np.zeros((len(starts), 1, config.mels, width), np.float32)
    for index, (start, size) in enumerate(zip(starts, valid)):
        windows[index, 0, :, :size] = spec[:, start:start + size]

    output = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
        sizes = torch.tensor(valid[start:start + batch_size], device=device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            tokens = backbone(input_values=batch, valid_timebins=sizes).last_hidden_state
            output.extend(head(tokens, sizes).sigmoid().float().cpu().numpy())
    kept = [output[0][:, :valid[0]]] if len(output) == 1 else [value[:, :750] if index == 0
        else value[:, 250:(valid[index] if index == len(output) - 1 else 750)]
        for index, value in enumerate(output)]
    return raw, np.concatenate(kept, axis=1)[:, :spec.shape[1]]


def main():
    parser = argparse.ArgumentParser(description="Plot BirdCODE annotations and SongMAE predictions on one spectrogram.")
    parser.add_argument("--dataset", choices=("wabad", "powdermill"), default="wabad")
    parser.add_argument("--root", type=Path, default=Path("data/birdcode/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/songmae-large-32x1-detector.pt"))
    parser.add_argument("--site", default="ARD", help="WABAD site")
    parser.add_argument("--recording", help="recording basename; defaults to the first recording")
    parser.add_argument("--start", type=float, default=0, help="start time of the five-second view")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/visualizations/wabad_vs_songmae.png"))
    args = parser.parse_args()

    selected = [Path(args.recording).stem] if args.recording else None
    rows = wabad(args.root, [args.site]) if args.dataset == "wabad" else powdermill(args.root, selected)
    row = next((item for item in rows if args.recording is None or Path(item[0]).name == Path(args.recording).name), None)
    assert row is not None, f"recording not found in {args.dataset}: {args.recording}"
    recording, waveform, annotations = row
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, head, _ = load_detector(args.checkpoint, device)
    config = backbone.config
    first = round(args.start * config.audio_sr)
    last = first + VIEW_SECONDS * config.audio_sr
    assert first >= 0 and last <= len(waveform), "five-second view extends beyond the recording"
    waveform = waveform[first:last]
    end = args.start + VIEW_SECONDS
    annotations = annotations[(annotations[:, 1] > args.start) & (annotations[:, 0] < end)].copy()
    annotations[:, 0] = np.maximum(annotations[:, 0], args.start)
    annotations[:, 1] = np.minimum(annotations[:, 1], end)
    raw, probability = predict(waveform, backbone, head, args.batch_size, device)

    times = args.start + np.arange(raw.shape[1]) * config.audio_hop_size / config.audio_sr
    frequencies = librosa.mel_frequencies(n_mels=config.mels, fmin=20, fmax=config.audio_sr // 2)
    smooth, detected = clean_mask(probability)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True, layout="constrained")
    for axis in axes:
        axis.pcolormesh(times, frequencies, raw, shading="nearest", cmap="magma")
        axis.set_ylabel("Frequency (Hz)")
    for onset, offset, low, high in annotations:
        axes[0].add_patch(Rectangle((onset, low), offset - onset, high - low,
            fill=False, edgecolor="cyan", linewidth=1.1))
    image = axes[1].pcolormesh(times, frequencies, np.ma.masked_where(~detected, smooth), shading="nearest",
        cmap="viridis", vmin=0, vmax=1, alpha=.7)
    if detected.any():
        axes[1].contour(times, frequencies, detected, levels=[.5], colors="white", linewidths=.8)
    name = "WABAD" if args.dataset == "wabad" else "Powdermill"
    axes[0].set_title(f"{name} annotations ({len(annotations)} boxes)")
    axes[1].set_title("SongMAE post-processed detections (white contours)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(args.start, end)
    fig.colorbar(image, ax=axes, label="Smoothed P(song)")
    fig.suptitle(f"{recording} · {args.start:g}–{end:g} s", fontsize=14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
