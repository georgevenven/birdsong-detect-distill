#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname) == george-server ]]
scaling_root=/media/george/DATA/songmae-scaling-seeds12-20260919
shared_root=/media/george/DATA/songmae-lr-sweep-20260916
cd "$scaling_root"
export HF_HOME="$shared_root/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$scaling_root/src:$scaling_root/scripts"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec "$shared_root/.venv/bin/python" -u scripts/run_v100_scaling_seeds.py >> driver.log 2>&1
