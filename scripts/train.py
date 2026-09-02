#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from birdsong_detect_distill.data import PixelWindows, read_rows, split_rows
from birdsong_detect_distill.model import DenseHead, load_backbone, loss


@torch.no_grad()
def evaluate(backbone, head, loader, data, device, tv_weight):
    head.eval()
    losses, rows, offset = [], [], 0
    for specs, targets, valid in loader:
        specs, targets = specs.to(device), targets.to(device)
        tokens = backbone(input_values=specs, valid_timebins=valid.to(device)).last_hidden_state
        logits = head(tokens.float(), valid)
        losses.append(float(loss(logits, targets, valid, True, tv_weight)))
        for index, length in enumerate(valid):
            length = int(length)
            rows.append((data.windows[offset + index][0]["recording"], targets[index, ::2, :length:4].cpu().numpy().ravel(),
                logits[index, ::2, :length:4].cpu().numpy().ravel()))
        offset += len(specs)
    return float(np.mean(losses)), rows


def best_threshold(rows):
    labels, scores = np.concatenate([x[1] for x in rows]), np.concatenate([x[2] for x in rows])
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1[:-1]))
    return float(thresholds[index]), float(precision[index]), float(recall[index]), float(f1[index])


def summary(rows, threshold):
    total, recordings = np.zeros(3, np.int64), defaultdict(lambda: np.zeros(3, np.int64))
    for recording, labels, scores in rows:
        predicted, truth = scores >= threshold, labels.astype(bool)
        value = np.asarray([(predicted & truth).sum(), (predicted & ~truth).sum(), (~predicted & truth).sum()])
        total += value
        recordings[recording] += value
    tp, fp, fn = total
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    macro = [2 * (tp / max(1, tp + fp)) * (tp / max(1, tp + fn)) /
        max(tp / max(1, tp + fp) + tp / max(1, tp + fn), 1e-12) for tp, fp, fn in recordings.values()]
    return {"precision": float(precision), "recall": float(recall),
        "micro_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "recording_macro_f1": float(np.mean(macro)), "counts": total.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Train a dense detector from Qwen boxes and frozen SongMAE.")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/xcl/qwen38_adaptive_review_5s_annotations.jsonl"))
    parser.add_argument("--shard-dir", type=Path, default=Path("data/xcl/shards"))
    parser.add_argument("--backbone", default="georgeven/songmae-large-32x1")
    parser.add_argument("--out", type=Path, default=Path("artifacts/songmae-large-32x1-detector.pt"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum", type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(args.backbone, device)
    rows = read_rows(args.annotations, args.seed, args.maximum)
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    train = PixelWindows(train_rows, args.shard_dir, backbone.config)
    val = PixelWindows(val_rows, args.shard_dir, backbone.config)
    config = backbone.config
    height, width = config.mels // config.patch_height, config.num_timebins // config.patch_width
    head = DenseHead(config.enc_hidden_d, args.hidden, height, width, config.patch_height, config.patch_width, args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=args.weight_decay)
    loaders = [DataLoader(data, batch, shuffle=index == 0, num_workers=2, pin_memory=True)
        for index, (data, batch) in enumerate(((train, args.batch_size), (val, args.eval_batch_size)))]
    best_state, best_loss = None, float("inf")
    print(f"{len(train)} train / {len(val)} validation windows; {sum(x.numel() for x in head.parameters()):,} trainable parameters")
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, (specs, targets, valid) in enumerate(loaders[0], 1):
            specs, targets = specs.to(device), targets.to(device)
            with torch.no_grad():
                tokens = backbone(input_values=specs, valid_timebins=valid.to(device)).last_hidden_state
            value = loss(head(tokens.float(), valid), targets, valid, True, args.tv_weight)
            (value / args.accumulation).backward()
            if step % args.accumulation == 0 or step == len(loaders[0]):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(value))
        torch.cuda.empty_cache()
        val_loss, validation = evaluate(backbone, head, loaders[1], val, device, args.tv_weight)
        score = best_threshold(validation)
        print(f"epoch {epoch}: train={np.mean(losses):.4f} val={val_loss:.4f} f1={score[3]:.3f} p={score[1]:.3f} r={score[2]:.3f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, {key: value.detach().cpu() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    _, validation = evaluate(backbone, head, loaders[1], val, device, args.tv_weight)
    recordings = sorted({x[0] for x in validation})
    np.random.default_rng(123).shuffle(recordings)
    calibration = set(recordings[:len(recordings) // 2])
    calibration_rows = [x for x in validation if x[0] in calibration]
    test_rows = [x for x in validation if x[0] not in calibration]
    threshold = best_threshold(calibration_rows)[0]
    report = {"best_val_loss": best_loss, "threshold": threshold, "calibration_recordings": len(calibration),
        "test_recordings": len(set(recordings) - calibration), "untouched_test": summary(test_rows, threshold)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "head": best_state, "hidden": args.hidden, "height": height, "width": width,
        "dropout": args.dropout, "backbone_id": args.backbone, "threshold": threshold, "tv_weight": args.tv_weight,
        "metrics": report}, args.out)
    print(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
