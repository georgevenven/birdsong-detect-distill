#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from birdsong_detect_distill.birdbox import model_path


def main():
    parser = argparse.ArgumentParser(description="Distill reviewed Qwen boxes into YOLO11n.")
    parser.add_argument("--data", type=Path, default=Path("data/yolo/qwen/dataset.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/yolo"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="0")
    parser.add_argument("--checkpoint", type=Path, help="New output checkpoint; never overwrite an existing file.")
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    summary_path = args.data.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    train_all = summary.get("train_all", False)
    name = "yolo11n-qwen-scratch.pt" if args.scratch else "yolo11n-qwen-birdbox-init.pt"
    checkpoint = args.checkpoint or args.out.parent / name
    if checkpoint.exists() or args.out.exists():
        raise ValueError("choose new checkpoint and training directories; existing runs are never overwritten")
    model = YOLO("yolo11n.yaml" if args.scratch else model_path("yolo11n", args.model_dir))
    model.train(data=args.data.resolve(), epochs=args.epochs, imgsz=1024, batch=args.batch_size,
        device=args.device, workers=8, project=args.out.resolve(), name="scratch" if args.scratch else "birdbox",
        patience=0 if train_all else 50, seed=0, deterministic=True, plots=False, val=not train_all)
    selected = model.trainer.last if train_all else model.trainer.best
    history = list(csv.DictReader(model.trainer.csv.open()))
    history = [{k.strip(): v for k, v in row.items()} for row in history]
    selected_row = history[-1] if train_all else max(reversed(history), key=lambda r: float(r["metrics/mAP50-95(B)"]))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, checkpoint)
    report = {"dataset": summary, "epochs": args.epochs, "batch_size": args.batch_size,
        "initialization": "random" if args.scratch else "released_birdbox_yolo11n",
        "selection": "fixed_final_epoch_no_validation_selection" if train_all else "best_validation_box_map",
        "selected_epoch": int(float(selected_row["epoch"])),
        "validation_selection_metric": None if train_all else "native_box_mAP50-95",
        "training_log_sha256": hashlib.sha256(model.trainer.csv.read_bytes()).hexdigest(),
        "training_args": str(model.trainer.save_dir / "args.yaml"),
        "training_args_sha256": hashlib.sha256((model.trainer.save_dir / "args.yaml").read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    checkpoint.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
