#!/usr/bin/env bash
set -euo pipefail
hexeberg_root=/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/yolo-hexeberg-25k-20260920
cd "$hexeberg_root"
export PYTHONPATH="$hexeberg_root/src:$hexeberg_root/scripts"
export YOLO_CONFIG_DIR="$hexeberg_root/ultralytics_config"
export MPLCONFIGDIR="$hexeberg_root/matplotlib_cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec "$hexeberg_root/venv/bin/python" -u scripts/run_hexeberg_yolo.py >> driver.log 2>&1
