#!/usr/bin/env bash
set -euo pipefail
study_root=/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/yolo-hexeberg-25k-20260920
cd "$study_root"
export PYTHONPATH="$study_root/src:$study_root/scripts"
export YOLO_CONFIG_DIR="$study_root/ultralytics_config" MPLCONFIGDIR="$study_root/matplotlib_cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export LD_PRELOAD="$study_root/migration_20260921/driver-extract-595.84/libnvidia-ml.so.595.84"
exec "$study_root/venv/bin/python" -u scripts/queue_hexeberg_migration.py >> migration_20260921/driver.log 2>&1
