#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
run=training_duration_10000train_800val_2026-09-10
labels=data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
mkdir -p "$artifacts" "$results/logs"
.venv/bin/python scripts/summarize_training_duration.py --prepare

lane() {
  local gpu=$1
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  local config size seed revision
  for config in "$@"; do
    size=${config%_*}
    seed=${config#*_}
    case "$size" in
      micro) revision=78a7f7e959e76d791d6ee6d8f761cb139c8654e4 ;;
      base) revision=d47970abb03ad01c575bd98ed21932283d00eaa0 ;;
      large) revision=f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e ;;
    esac
    if [[ ! -f "$artifacts/seed$seed/$size.pt" ]]; then
      .venv/bin/python -u scripts/train.py \
        --annotations "$labels/train/self_review.jsonl" --validation-annotations "$labels/validation/self_review.jsonl" \
        --backbone "georgeven/songmae-$size-32x1" --backbone-revision "$revision" \
        --epochs 15 --batch-size 16 --eval-batch-size 4 --accumulation 1 --seed "$seed" \
        --hidden 128 --dropout 0.1 --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
        --log-every 25 --out "$artifacts/seed$seed/$size.pt" \
        2>&1 | tee -a "$results/logs/train_${size}_seed$seed.log"
    fi
    .venv/bin/python -u scripts/evaluate_current_backbones.py --size "$size" --seconds 10000 \
      --seed "$seed" --training-epochs 15 --prediction-cache samples \
      2>&1 | tee -a "$results/logs/evaluate_${size}_seed$seed.log"
  done
}

lane GPU-93189925-4747-5857-e01f-d44d77ad07a7 large_0 base_1 micro_1 large_2 & first=$!
lane GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556 base_0 micro_0 large_1 base_2 micro_2 & second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
if (( status != 0 )); then exit "$status"; fi
.venv/bin/python scripts/summarize_training_duration.py
