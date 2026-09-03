#!/usr/bin/env python3
import argparse
import json
from itertools import batched, islice
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from birdsong_detect_distill.evaluation import WABAD_SITES, average_precision, powdermill, wabad, xcsl
from birdsong_detect_distill.model import clean_mask, load_detector
from evaluate_qwen_yolo import windows


THRESHOLDS = np.linspace(0, 1, 51)


def references(events):
    events = np.asarray(events).reshape(-1, 4)
    low = librosa.hz_to_mel(20)
    scale = 128 / (librosa.hz_to_mel(16000) - low)
    return np.column_stack((events[:, 0], (librosa.hz_to_mel(np.clip(events[:, 2], 20, 16000)) - low) * scale,
        events[:, 1], (librosa.hz_to_mel(np.clip(events[:, 3], 20, 16000)) - low) * scale))


def match(reference, estimate, iou):
    if not len(reference) or not len(estimate):
        return np.asarray([0, len(estimate), len(reference)])
    left = np.maximum(reference[:, None, :2], estimate[None, :, :2])
    right = np.minimum(reference[:, None, 2:], estimate[None, :, 2:])
    intersection = np.maximum(0, right - left).prod(2)
    reference_area = np.maximum(0, reference[:, 2:] - reference[:, :2]).prod(1)
    estimate_area = np.maximum(0, estimate[:, 2:] - estimate[:, :2]).prod(1)
    valid = intersection / np.maximum(reference_area[:, None] + estimate_area[None] - intersection, 1e-12) >= iou
    row, column = linear_sum_assignment(valid, maximize=True)
    true = int(valid[row, column].sum())
    return np.asarray([true, len(estimate) - true, len(reference) - true])


def yolo_boxes(model, waveform, device):
    _, views = windows(waveform)
    results = []
    for chunk in batched([x[2] for x in views], 16):
        results.extend(model.predict(list(chunk), imgsz=1024, device=device, conf=.001, iou=.7,
            batch=len(chunk), verbose=False))
    boxes = []
    for index, (result, (start, valid, _)) in enumerate(zip(results, views)):
        owner_left, owner_right = (0 if index == 0 else 250), (valid if index == len(views) - 1 else 750)
        for box, score in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            if owner_left <= (box[0] + box[2]) * 500 < owner_right:
                boxes.append([(start + box[0] * 1000) / 200, (1 - box[3]) * 128,
                    (start + box[2] * 1000) / 200, (1 - box[1]) * 128, score])
    return np.asarray(boxes).reshape(-1, 5)


@torch.inference_mode()
def songmae_probability(backbone, head, waveform, device):
    config = backbone.config
    spec = librosa.feature.melspectrogram(y=waveform, sr=32000, n_fft=1024,
        hop_length=160, power=2, n_mels=128, fmin=20, fmax=16000)
    spec = (librosa.power_to_db(spec, ref=np.max, top_db=None).astype(np.float32) - config.audio_mean) / config.audio_std
    width, hop = 1000, 500
    starts = list(range(0, max(1, spec.shape[1] - width + hop), hop))
    if starts[-1] + width < spec.shape[1]:
        starts.append(starts[-1] + hop)
    windows_, valid = [], []
    for start in starts:
        size = min(width, spec.shape[1] - start)
        windows_.append(np.pad(spec[:, start:start + size], ((0, 0), (0, width - size))))
        valid.append(size)
    output = []
    for indexes in batched(range(len(windows_)), 4):
        indexes = list(indexes)
        values = torch.from_numpy(np.asarray([windows_[i] for i in indexes]))[:, None].to(device)
        lengths = torch.tensor([valid[i] for i in indexes], device=device)
        with torch.autocast(device.type):
            tokens = backbone(input_values=values, valid_timebins=lengths).last_hidden_state
            output.extend(head(tokens, lengths).sigmoid().float().cpu().numpy())
    probability = np.zeros((128, spec.shape[1]), np.float32)
    for value, start, size in zip(output, starts, valid):
        probability[:, start:start + size] = np.maximum(probability[:, start:start + size], value[:, :size])
    logits = np.log(np.clip(probability, 1e-5, 1 - 1e-5) / np.clip(1 - probability, 1e-5, 1))
    return 1 / (1 + np.exp(-ndimage.gaussian_filter(logits, (2, 3))))


def songmae_boxes(probability):
    smooth, mask = clean_mask(probability)
    labels, count = ndimage.label(mask)
    boxes = []
    for component in range(1, count + 1):
        y, x = np.where(labels == component)
        boxes.append([x.min() / 200, y.min(), (x.max() + 1) / 200, y.max() + 1, smooth[y, x].max()])
    return np.asarray(boxes).reshape(-1, 5)


def summarize(counts):
    output = {f"ap_iou_{iou}": average_precision(value) for iou, value in counts.items()}
    for iou, value in counts.items():
        tp, fp, fn = value.T
        precision, recall = tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        index = int(np.argmax(f1))
        output[f"best_f1_iou_{iou}"] = {"f1": float(f1[index]), "precision": float(precision[index]),
            "recall": float(recall[index]), "threshold": float(THRESHOLDS[index])}
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO and SongMAE using human two-dimensional boxes.")
    parser.add_argument("--dataset", choices=("powdermill", "xcsl", "wabad"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("results/reproduced"))
    args = parser.parse_args()
    yolo = YOLO("artifacts/yolo11n-qwen-birdbox-init.pt")
    device = torch.device("cuda:1")
    backbone, head, _ = load_detector("artifacts/songmae-large-32x1-detector.pt", device)
    if args.dataset == "wabad":
        stride = max(1, len(WABAD_SITES) // args.maximum)
        source = (row for site in WABAD_SITES[::stride][:args.maximum] for row in islice(wabad(args.root, [site]), 1))
    else:
        source = {"powdermill": powdermill, "xcsl": xcsl}[args.dataset](args.root)
        source = islice(source, args.maximum)
    counts = {model: {iou: np.zeros((len(THRESHOLDS), 3), np.int64) for iou in (.2, .5)}
        for model in ("yolo11n_qwen", "songmae_large_32x1")}
    files = 0
    for recording, waveform, events in source:
        reference = references(events)
        predicted = yolo_boxes(yolo, waveform, "cuda:0")
        probability = songmae_probability(backbone, head, waveform, device)
        songmae = songmae_boxes(probability)
        for index, threshold in enumerate(THRESHOLDS):
            estimates = {"yolo11n_qwen": predicted[predicted[:, 4] >= threshold, :4],
                "songmae_large_32x1": songmae[songmae[:, 4] >= threshold, :4]}
            for model, boxes in estimates.items():
                for iou in (.2, .5):
                    counts[model][iou][index] += match(reference, boxes, iou)
        files += 1
        print(f"{args.dataset}: {files}", flush=True)
    result = {"dataset": args.dataset, "files": files, "coordinate_space": "time_seconds_x_mel_frequency",
        "thresholds": len(THRESHOLDS), "models": {model: summarize(value) for model, value in counts.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"spatial_2d_{args.dataset}_{files}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
