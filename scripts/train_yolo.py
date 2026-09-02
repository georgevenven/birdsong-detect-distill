#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from evaluate_birdbox import model_path


def main():
    parser = argparse.ArgumentParser(description="Distill reviewed Qwen boxes into YOLO11n.")
    parser.add_argument("--data", type=Path, default=Path("data/yolo/qwen/dataset.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/yolo"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    model = YOLO("yolo11n.yaml" if args.scratch else model_path("yolo11n", args.model_dir))
    model.train(data=args.data.resolve(), epochs=args.epochs, imgsz=1024, batch=args.batch_size,
        device=args.device, workers=8, project=args.out.resolve(), name="scratch" if args.scratch else "birdbox",
        patience=10, seed=0, deterministic=True, plots=False)
    name = "yolo11n-qwen-scratch.pt" if args.scratch else "yolo11n-qwen-birdbox-init.pt"
    shutil.copy2(model.trainer.best, args.out.parent / name)


if __name__ == "__main__":
    main()
