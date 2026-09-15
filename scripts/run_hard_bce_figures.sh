#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
artifacts=artifacts/hard_bce_figures_2026-09-08
results=results/qwen_teacher_powdermill/hard_bce_figures_2026-09-08
for config in large_s100 large_s1000 micro_s10000 base_s10000; do
  size=${config%_s*}
  seconds=${config#*_s}
  case "$size" in
    micro) revision=78a7f7e959e76d791d6ee6d8f761cb139c8654e4 ;;
    base) revision=d47970abb03ad01c575bd98ed21932283d00eaa0 ;;
    large) revision=f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e ;;
  esac
  .venv/bin/python -u scripts/train.py \
    --annotations "data/annotations/xcl/ablation_2500/ap_10000/${seconds}s/self_review.jsonl" \
    --backbone "georgeven/songmae-$size-32x1" --backbone-revision "$revision" \
    --train-all --epochs 3 --batch-size 16 --seed 0 --hidden 128 --dropout 0.1 \
    --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
    --out "$artifacts/$config.pt" 2>&1 | tee "$results/logs/train_$config.log"
done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
.venv/bin/python -u scripts/evaluate_hard_bce_figures.py 2>&1 | tee "$results/logs/evaluate.log"
