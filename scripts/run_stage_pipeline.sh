#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
run=stage_pipeline_10000train_800val_2026-09-09
labels="data/annotations/xcl/$run"
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
mkdir -p "$artifacts" "$results/logs"

train_lane() {
  local lane=$1
  local variants
  if [[ "$lane" == 0 ]]; then
    export CUDA_VISIBLE_DEVICES=GPU-93189925-4747-5857-e01f-d44d77ad07a7
    variants=(reasoning shifted_review)
  else
    export CUDA_VISIBLE_DEVICES=GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556
    variants=(self_review full)
  fi
  for variant in "${variants[@]}"; do
    if [[ -f "$artifacts/$variant.pt" ]]; then continue; fi
    .venv/bin/python -u scripts/train.py \
      --annotations "$labels/train/$variant.jsonl" \
      --validation-annotations "$labels/validation/$variant.jsonl" \
      --shard-dir data/xcl/shards --backbone georgeven/songmae-large-32x1 \
      --backbone-revision f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e \
      --epochs 3 --batch-size 16 --eval-batch-size 4 --accumulation 1 --seed 0 \
      --hidden 128 --dropout 0.1 --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
      --log-every 25 --out "$artifacts/$variant.pt" 2>&1 | tee -a "$results/logs/train_$variant.log"
  done
}

train_lane 0 & lane0=$!
train_lane 1 & lane1=$!
status=0
wait "$lane0" || status=1
wait "$lane1" || status=1
if (( status != 0 )); then exit "$status"; fi

export CUDA_VISIBLE_DEVICES=GPU-93189925-4747-5857-e01f-d44d77ad07a7
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
.venv/bin/python -u scripts/evaluate_stage_pipeline.py \
  --root /home/george-vengrovski/Documents/SongMAE/data/birdcode/raw \
  --split "$labels/manifest.json" --checkpoints "$artifacts" \
  --teacher-results results/qwen_teacher_powdermill/scores.json \
  --cache "$artifacts/predictions" --out "$results" 2>&1 | tee -a "$results/logs/evaluate.log"
