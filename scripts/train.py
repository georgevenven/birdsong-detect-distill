#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, RandomSampler

from birdsong_detect_distill.data import PixelWindows, read_rows, split_rows
from birdsong_detect_distill.model import DenseHead, load_backbone, loss, supervision_mask


class RecordedSampler:
    def __init__(self, data):
        self.sampler = RandomSampler(data)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        self.digest = hashlib.sha256()
        for index in self.sampler:
            self.digest.update(int(index).to_bytes(8, "little"))
            yield index


@torch.no_grad()
def evaluate(backbone, head, loader, data, device, tv_weight, target_smoothing=True, collect_rows=True, ignore_uncertain=False):
    head.eval()
    losses, weights, rows, offset = [], [], [], 0
    for specs, targets, valid in loader:
        specs, targets = specs.to(device), targets.to(device)
        weight = int(supervision_mask(targets, valid).sum()) if ignore_uncertain else (1 if collect_rows else int(valid.sum()))
        if weight == 0:
            continue
        tokens = backbone(input_values=specs, valid_timebins=valid.to(device)).last_hidden_state
        logits = head(tokens.float(), valid)
        losses.append(float(loss(logits, targets, valid, target_smoothing, tv_weight, ignore_uncertain)))
        weights.append(weight)
        if not collect_rows:
            continue
        for index, length in enumerate(valid):
            length = int(length)
            rows.append((data.windows[offset + index][0]["recording"], targets[index, ::2, :length:4].cpu().numpy().ravel(),
                logits[index, ::2, :length:4].cpu().numpy().ravel()))
        offset += len(specs)
    if not weights:
        raise ValueError("validation has no supervised pixels")
    return float(np.average(losses, weights=weights)), rows


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
    parser.add_argument("--validation-annotations", type=Path,
        help="Explicit recording-disjoint validation labels for checkpoint selection, not threshold calibration.")
    parser.add_argument("--shard-dir", type=Path, default=Path("data/xcl/shards"))
    parser.add_argument("--backbone", default="georgeven/songmae-large-32x1")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--out", type=Path, default=Path("artifacts/songmae-large-32x1-detector.pt"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument("--ignore-uncertain", action="store_true",
        help="Ignore uncertain-only pixels in training and explicit XC validation; target/chorus overlaps stay positive.")
    parser.add_argument("--confidence-targets", action="store_true",
        help="Use maximum foreground-box confidence as the BCE target; zero outside boxes, including all foreground labels.")
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--target-smoothing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=.25)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recordings", type=int)
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.head_layers < 0:
        parser.error("--head-layers must be nonnegative")
    if args.ignore_uncertain and (not args.validation_annotations or args.target_smoothing or args.tv_weight != 0 or args.accumulation != 1):
        parser.error("--ignore-uncertain requires explicit validation, hard BCE, TV=0, and accumulation=1")
    if args.confidence_targets and (args.ignore_uncertain or not args.validation_annotations or args.target_smoothing or args.tv_weight != 0):
        parser.error("--confidence-targets requires explicit validation, no ignored uncertainty, no spatial smoothing, and TV=0")
    if args.out.exists():
        raise ValueError("checkpoint already exists; choose a new output path")
    if args.validation_annotations and (args.train_all or args.recordings or args.maximum):
        parser.error("explicit validation cannot be combined with --train-all, --recordings or --maximum")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(args.backbone, device, revision=args.backbone_revision)
    rows = read_rows(args.annotations, args.seed, args.maximum)
    selected = sorted({row["recording"] for row in rows})
    random.Random(args.seed).shuffle(selected)
    if args.recordings:
        selected = selected[:args.recordings]
        keep = set(selected)
        rows = [row for row in rows if row["recording"] in keep]
    if args.validation_annotations:
        train_rows, val_rows = rows, read_rows(args.validation_annotations, args.seed)
        if not train_rows or not val_rows:
            raise ValueError("training and validation must both contain annotations")
        overlap = {r["recording"].upper() for r in train_rows} & {r["recording"].upper() for r in val_rows}
        if overlap:
            raise ValueError(f"training/validation recording overlap: {sorted(overlap)}")
    else:
        train_rows, val_rows = (rows, []) if args.train_all else split_rows(rows, args.val_fraction, args.seed)
    train = PixelWindows(train_rows, args.shard_dir, backbone.config, args.ignore_uncertain, args.confidence_targets)
    config = backbone.config
    height, width = config.mels // config.patch_height, config.num_timebins // config.patch_width
    head = DenseHead(config.enc_hidden_d, args.hidden, height, width, config.patch_height, config.patch_width,
        args.dropout, args.head_layers).to(device)
    initial_head_sha256 = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
        for value in head.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(head.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    sampler = RecordedSampler(train)
    train_loader = DataLoader(train, args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    if not args.train_all:
        val = PixelWindows(val_rows, args.shard_dir, backbone.config, args.ignore_uncertain, args.confidence_targets)
        val_loader = DataLoader(val, args.eval_batch_size, num_workers=2, pin_memory=True)
    supervision = {"train": train.supervision_counts(), "validation": val.supervision_counts()} if args.ignore_uncertain else None
    if supervision:
        print(f"Ignoring uncertain-only pixels: {json.dumps(supervision)}", flush=True)
    if args.confidence_targets:
        print("BCE targets: maximum foreground-box confidence, zero background; no loss weighting or ignored labels.", flush=True)
    best_state, best_loss, best_epoch = None, float("inf"), None
    history = []
    print(f"{len(train)} train / {len(val_rows)} validation windows; {sum(x.numel() for x in head.parameters()):,} trainable parameters")
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, (specs, targets, valid) in enumerate(train_loader, 1):
            specs, targets = specs.to(device), targets.to(device)
            if args.ignore_uncertain and not supervision_mask(targets, valid).any():
                continue
            with torch.no_grad():
                tokens = backbone(input_values=specs, valid_timebins=valid.to(device)).last_hidden_state
            value = loss(head(tokens.float(), valid), targets, valid, args.target_smoothing, args.tv_weight, args.ignore_uncertain)
            (value / args.accumulation).backward()
            if step % args.accumulation == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(value))
            if args.log_every and step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)}: train={np.mean(losses):.4f}", flush=True)
        history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)),
            "sample_order_sha256": sampler.digest.hexdigest(),
            "cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()})
        if args.train_all:
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            print(f"epoch {epoch}: train={np.mean(losses):.4f}", flush=True)
            continue
        torch.cuda.empty_cache()
        val_loss, validation = evaluate(backbone, head, val_loader, val, device, args.tv_weight,
            args.target_smoothing, collect_rows=not args.validation_annotations, ignore_uncertain=args.ignore_uncertain)
        history[-1]["validation_loss"] = val_loss
        if args.validation_annotations:
            print(f"epoch {epoch}: train={np.mean(losses):.4f} val={val_loss:.4f}", flush=True)
        else:
            score = best_threshold(validation)
            print(f"epoch {epoch}: train={np.mean(losses):.4f} val={val_loss:.4f} f1={score[3]:.3f} p={score[1]:.3f} r={score[2]:.3f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if best_state is None:
        raise ValueError("no finite validation loss or completed training epoch")
    head.load_state_dict(best_state)
    if args.validation_annotations:
        threshold = args.threshold if args.threshold is not None else 0.0
        report = {"recordings": len(selected), "rows": len(rows), "train_windows": len(train),
            "validation_windows": len(val), "validation_recordings": len({r["recording"] for r in val_rows}),
            "train_timebins": sum(w[2] for w in train.windows), "validation_timebins": sum(w[2] for w in val.windows),
            "epochs": args.epochs, "selected_epoch": best_epoch, "best_val_loss": best_loss,
            "checkpoint_selection": "minimum_recording_disjoint_validation_loss",
            "threshold_source": "provided" if args.threshold is not None else "external_calibration_required"}
    elif args.train_all:
        threshold = args.threshold if args.threshold is not None else 0.0
        report = {"recordings": len(selected), "rows": len(rows), "train_windows": len(train),
            "epochs": args.epochs, "threshold_source": "provided" if args.threshold is not None else "uncalibrated"}
    else:
        _, validation = evaluate(backbone, head, val_loader, val, device, args.tv_weight, args.target_smoothing)
        recordings = sorted({x[0] for x in validation})
        np.random.default_rng(123).shuffle(recordings)
        calibration = set(recordings[:len(recordings) // 2])
        calibration_rows = [x for x in validation if x[0] in calibration]
        test_rows = [x for x in validation if x[0] not in calibration]
        threshold = best_threshold(calibration_rows)[0]
        report = {"best_val_loss": best_loss, "threshold": threshold, "calibration_recordings": len(calibration),
            "test_recordings": len(set(recordings) - calibration), "untouched_test": summary(test_rows, threshold)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "head": best_state, "hidden": args.hidden, "head_layers": args.head_layers,
        "height": height, "width": width,
        "dropout": args.dropout, "backbone_id": args.backbone, "threshold": threshold, "tv_weight": args.tv_weight,
        "backbone_revision": getattr(backbone.config, "_commit_hash", None),
        "target_smoothing": args.target_smoothing, "seed": args.seed, "initial_head_sha256": initial_head_sha256,
        "ignore_uncertain": args.ignore_uncertain, "supervision": supervision,
        "confidence_targets": args.confidence_targets,
        "training_history": history,
        "training_config": {"optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "accumulation": args.accumulation, "train_all": args.train_all,
            "target_smoothing_mel_time": [31, 3] if args.target_smoothing else None},
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "validation_annotations_sha256": hashlib.sha256(args.validation_annotations.read_bytes()).hexdigest()
            if args.validation_annotations else None,
        "training_recording_ids": sorted({r["recording"] for r in train_rows}),
        "validation_recording_ids": sorted({r["recording"] for r in val_rows}),
        "metrics": report}, args.out)
    print(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
