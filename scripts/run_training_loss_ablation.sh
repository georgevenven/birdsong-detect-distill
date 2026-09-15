#!/usr/bin/env bash
# Two independent GPU lanes; identical data, initialization, order and epoch budget.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
run=training_loss_ablation_2026-09-08
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
lane=$1
case "$lane" in
  0) export CUDA_VISIBLE_DEVICES=GPU-93189925-4747-5857-e01f-d44d77ad07a7; tv=0.001; suffix=tv ;;
  1) export CUDA_VISIBLE_DEVICES=GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556; tv=0; suffix=no_tv ;;
  *) exit 2 ;;
esac
for target in soft hard; do
  smoothing=--target-smoothing
  if [[ "$target" == hard ]]; then smoothing=--no-target-smoothing; fi
  .venv/bin/python scripts/train.py \
    --annotations data/annotations/xcl/ablation_2500/ap_10000/10000s/self_review.jsonl \
    --backbone georgeven/songmae-large-32x1 \
    --backbone-revision f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e \
    --train-all --epochs 3 --batch-size 16 --accumulation 1 --seed 0 \
    --hidden 128 --dropout 0.1 --weight-decay 0.0001 --tv-weight "$tv" "$smoothing" \
    --out "$artifacts/${target}_${suffix}.pt" \
    2>&1 | tee "$results/logs/${target}_${suffix}.log"
done
