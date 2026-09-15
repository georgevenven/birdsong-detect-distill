"""BirdBox 094d5c3 preprocessing, including DSP.jl's implicit FFT padding."""
import hashlib
from itertools import batched
from pathlib import Path

import numpy as np
import requests
from matplotlib import colormaps
from scipy import interpolate, signal

REVISION = "094d5c36abda3d7f29151a2dca6d00282cbfbc8d"
MODELS = {
    "yolo11n": ("87abf5c37f225f4a24df2a18c9bd64e6caf8c464a3730d413221909bf1599051", 2.6),
    "yolo11l": ("8806e5d798acc3f61b435e18114da284aa8fb0dfdc6abe2d71b349d1da024e17", 25.3),
}
MAGMA = np.asarray(colormaps["magma"].colors)


def model_path(name, directory):
    path = Path(directory) / f"{name}.pt"
    expected = MODELS[name][0]
    if not path.exists():
        response = requests.get(f"https://raw.githubusercontent.com/org-arl/birdwatch-public/{REVISION}/birdbox/models/{name}.pt", timeout=120)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError("BirdBox download checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {path}")
    return path


def nextfastfft(n):
    # DSP.jl uses factors 2,3,5,7; scipy.fft.next_fast_len can also use 11.
    for size in range(n, 2 * n):
        remainder = size
        for prime in (2, 3, 5, 7):
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return size
    raise ValueError(n)


def spectrogram_image(waveform, sample_rate=32000, duration=6):
    length = max(2, round(sample_rate / (44100 / 4096)))
    hop = max(1, int(np.floor(sample_rate * duration / 1024)) + 1)
    waveform = np.pad(np.asarray(waveform, np.float64), length // 2)
    frequency, _, power = signal.spectrogram(waveform, sample_rate, signal.windows.hann(length, sym=True),
        length, length - hop, nextfastfft(length), detrend=False, scaling="density", mode="psd")
    # Match Julia's fs/nfft multiplication order at exact 500/12000-Hz boundaries.
    frequency = np.arange(len(frequency)) * (sample_rate / nextfastfft(length))
    power = 10 * np.log10(np.maximum(power / max(power.max(), np.finfo(float).eps), 1e-8))
    power = power[(frequency >= 500) & (frequency <= 12000)]
    if len(power) < 2:
        raise ValueError("audio has insufficient frequency coverage for BirdBox")
    power = interpolate.interp1d(np.linspace(0, 1, len(power)), power, axis=0)(np.linspace(0, 1, 1024))
    low, high = np.percentile(power, (1, 99.8))
    value = np.clip((np.clip(power, low, high) - low) / max(high - low, np.finfo(float).eps), 0, 1) ** .85
    # ColorSchemes.get linearly interpolates RGB; RGB{N0f8} rounds, not truncates.
    position = value[::-1] * 255
    left = np.floor(position).astype(int)
    mix = (position - left)[..., None]
    rgb = MAGMA[left] * (1 - mix) + MAGMA[np.minimum(left + 1, 255)] * mix
    return np.ascontiguousarray(np.rint(rgb * 255).astype(np.uint8)[..., ::-1])


def windows(waveform, sample_rate=32000, duration=6, overlap=1):
    length = round(duration * sample_rate)
    for start in np.arange(0, max(0, len(waveform) / sample_rate - 1e-9) + 1e-12, duration - overlap):
        index = round(start * sample_rate)
        clip = waveform[index:index + length]
        yield float(start), np.pad(clip, (0, length - len(clip)))


def predict_boxes(model, waveform, sample_rate, device, confidence=.0001, batch_size=8, max_det=3000):
    boxes = []
    for batch in batched(windows(waveform, sample_rate), batch_size):
        results = model.predict([spectrogram_image(x, sample_rate) for _, x in batch], imgsz=1024,
            device=device, conf=confidence, iou=.7, max_det=max_det, verbose=False)
        for result, (start, _) in zip(results, batch):
            if len(result.boxes) >= max_det:
                raise ValueError("YOLO hit max_det; increase it before reporting AP")
            for (left, top, right, bottom), score in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                boxes.append([start + left * 6, min(len(waveform) / sample_rate, start + right * 6),
                    500 + (1 - bottom) * 11500, 500 + (1 - top) * 11500, float(score)])
    return np.asarray([b for b in boxes if b[1] > b[0]], np.float32).reshape(-1, 5)
