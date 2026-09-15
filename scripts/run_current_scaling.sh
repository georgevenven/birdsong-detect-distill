#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
run=scaling_10000train_800val_2026-09-10
labels="data/annotations/xcl/$run"
validation=data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09/validation/self_review.jsonl
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
mkdir -p "$artifacts" "$results/logs"

lane() {
  local seconds=$1 gpu=$2
  export CUDA_VISIBLE_DEVICES="$gpu"
  if [[ ! -f "$artifacts/large_s$seconds.pt" ]]; then
    .venv/bin/python -u scripts/train.py \
      --annotations "$labels/self_review_s$seconds.jsonl" --validation-annotations "$validation" \
      --backbone georgeven/songmae-large-32x1 --backbone-revision f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e \
      --epochs 3 --batch-size 16 --eval-batch-size 4 --accumulation 1 --seed 0 \
      --hidden 128 --dropout 0.1 --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
      --log-every 10 --out "$artifacts/large_s$seconds.pt" 2>&1 | tee -a "$results/logs/train_s$seconds.log"
  fi
  .venv/bin/python -u scripts/evaluate_current_backbones.py --size large --seconds "$seconds" \
    2>&1 | tee -a "$results/logs/evaluate_s$seconds.log"
}

lane 100 GPU-93189925-4747-5857-e01f-d44d77ad07a7 & small=$!
lane 1000 GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556 & medium=$!
status=0
wait "$small" || status=1
wait "$medium" || status=1
if (( status != 0 )); then exit "$status"; fi
.venv/bin/python scripts/summarize_current_scaling.py
.venv/bin/python scripts/plot_combined_scaling.py \
  --scaling "$results/scaling.json" --backbones "$results/backbones.json" \
  --legend-position top --out "$results/xc_scaling_and_backbones" \
  2>&1 | tee -a "$results/logs/plot.log"
