#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
run=backbones_10000train_800val_2026-09-10
labels=data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
mkdir -p "$artifacts" "$results/logs"

lane() {
  local size=$1 gpu=$2 revision=$3
  export CUDA_VISIBLE_DEVICES="$gpu"
  if [[ ! -f "$artifacts/$size.pt" ]]; then
    .venv/bin/python -u scripts/train.py \
      --annotations "$labels/train/self_review.jsonl" \
      --validation-annotations "$labels/validation/self_review.jsonl" \
      --backbone "georgeven/songmae-$size-32x1" --backbone-revision "$revision" \
      --epochs 3 --batch-size 16 --eval-batch-size 4 --accumulation 1 --seed 0 \
      --hidden 128 --dropout 0.1 --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
      --log-every 25 --out "$artifacts/$size.pt" 2>&1 | tee -a "$results/logs/train_$size.log"
  fi
  .venv/bin/python -u scripts/evaluate_current_backbones.py --size "$size" \
    2>&1 | tee -a "$results/logs/evaluate_$size.log"
}

lane micro GPU-93189925-4747-5857-e01f-d44d77ad07a7 78a7f7e959e76d791d6ee6d8f761cb139c8654e4 & micro=$!
lane base GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556 d47970abb03ad01c575bd98ed21932283d00eaa0 & base=$!
status=0
wait "$micro" || status=1
wait "$base" || status=1
if (( status != 0 )); then exit "$status"; fi
.venv/bin/python scripts/plot_current_backbones.py 2>&1 | tee -a "$results/logs/plot.log"
