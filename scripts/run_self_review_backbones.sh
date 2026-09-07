#!/usr/bin/env bash
set -euo pipefail

python_bin=${PYTHON_BIN:-.venv/bin/python}
birdcode_root=${BIRDCODE_ROOT:-/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw}
annotations=data/annotations/xcl/ablation_2500/ap_10000/10000s/self_review.jsonl
artifact_dir=artifacts/backbone_self_review_10000s_2026-09-07
result_dir=results/qwen_teacher_powdermill/backbone_self_review_10000s_2026-09-07
teacher=results/qwen_teacher_powdermill/partial_1677_windows_2026-09-07.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$artifact_dir" "$result_dir/logs"

for size in micro base large; do
  checkpoint="$artifact_dir/$size-self_review-s10000.pt"
  if [[ "$size" == large ]]; then
    checkpoint=artifacts/qwen_ablation_ap_10000/self_review-s10000.pt
    [[ -f "$checkpoint" ]]
  elif [[ ! -f "$checkpoint" ]]; then
    "$python_bin" -u scripts/train.py --annotations "$annotations" --shard-dir data/xcl/shards \
      --backbone "georgeven/songmae-$size-32x1" --train-all --epochs 3 --batch-size 16 --seed 0 \
      --hidden 128 --dropout .1 --weight-decay 1e-4 --tv-weight 1e-3 --out "$checkpoint" \
      2>&1 | tee "$result_dir/logs/train-$size.log"
  fi
  if [[ ! -f "$result_dir/$size/comparison.json" ]]; then
    "$python_bin" -u scripts/compare_qwen_songmae.py --teacher-results "$teacher" \
      --variant self_review --checkpoint "$checkpoint" --root "$birdcode_root" --device cuda:0 \
      --out "$result_dir/$size" 2>&1 | tee "$result_dir/logs/evaluate-$size.log"
  fi
done
