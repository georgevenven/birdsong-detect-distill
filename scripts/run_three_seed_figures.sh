#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
run=scaling_backbones_three_seeds_2026-09-10
labels=data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09
small_labels=data/annotations/xcl/scaling_10000train_800val_2026-09-10
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
mkdir -p "$artifacts" "$results/logs"
.venv/bin/python scripts/summarize_three_seed_figures.py --prepare

lane() {
  local seed=$1 gpu=$2
  local config size seconds revision train_labels
  export CUDA_VISIBLE_DEVICES="$gpu"
  for config in large_s100 large_s1000 large_s10000 micro_s10000 base_s10000; do
    size=${config%_s*}
    seconds=${config#*_s}
    case "$size" in
      micro) revision=78a7f7e959e76d791d6ee6d8f761cb139c8654e4 ;;
      base) revision=d47970abb03ad01c575bd98ed21932283d00eaa0 ;;
      large) revision=f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e ;;
    esac
    train_labels="$labels/train/self_review.jsonl"
    if (( seconds != 10000 )); then train_labels="$small_labels/self_review_s$seconds.jsonl"; fi
    if [[ ! -f "$artifacts/seed$seed/$config.pt" ]]; then
      .venv/bin/python -u scripts/train.py \
        --annotations "$train_labels" --validation-annotations "$labels/validation/self_review.jsonl" \
        --backbone "georgeven/songmae-$size-32x1" --backbone-revision "$revision" \
        --epochs 3 --batch-size 16 --eval-batch-size 4 --accumulation 1 --seed "$seed" \
        --hidden 128 --dropout 0.1 --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
        --log-every 25 --out "$artifacts/seed$seed/$config.pt" \
        2>&1 | tee -a "$results/logs/train_seed${seed}_$config.log"
    fi
    .venv/bin/python -u scripts/evaluate_current_backbones.py --size "$size" --seconds "$seconds" \
      --seed "$seed" --prediction-cache samples \
      2>&1 | tee -a "$results/logs/evaluate_seed${seed}_$config.log"
  done
}

lane 1 GPU-93189925-4747-5857-e01f-d44d77ad07a7 & first=$!
lane 2 GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556 & second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
if (( status != 0 )); then exit "$status"; fi
.venv/bin/python scripts/summarize_three_seed_figures.py
.venv/bin/python scripts/plot_combined_scaling.py \
  --scaling "$results/scaling.json" --backbones "$results/backbones.json" \
  --legend-position top --out "$results/xc_scaling_and_backbones" \
  2>&1 | tee -a "$results/logs/plot.log"
