#!/usr/bin/env python3
"""Teacher-trained YOLO uses its training-time mel/viridis representation, not BirdBox's."""
import librosa
import numpy as np

from birdsong_detect_distill.qwen import image


def spectrogram(waveform):
    value = librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=1024,
        hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000)
    return librosa.power_to_db(value, ref=np.max, top_db=None).astype(np.float32)


def windows(waveform):
    spec, width, hop = spectrogram(waveform), 1000, 500
    starts = list(range(0, max(1, spec.shape[1] - width + hop), hop))
    if starts[-1] + width < spec.shape[1]:
        starts.append(starts[-1] + hop)
    output = []
    for start in starts:
        valid = min(width, spec.shape[1] - start)
        window = np.pad(spec[:, start:start + valid], ((0, 0), (0, width - valid)), constant_values=-100)
        output.append((start, valid, np.ascontiguousarray(np.asarray(image(window))[:, :, ::-1])))
    return spec.shape[1], output


if __name__ == "__main__":
    from evaluate_baselines import main
    main(default_model="qwen_yolo")
