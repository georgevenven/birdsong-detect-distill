#!/usr/bin/env bash
set -euo pipefail

birdcode_root=${BIRDCODE_ROOT:-/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw}
python_bin=${PYTHON_BIN:-.venv/bin/python}
subset_dir=data/annotations/xcl/ablation_2500/preliminary
artifact_dir=artifacts/qwen_ablation_preliminary
result_dir=results/qwen_ablation_preliminary

"$python_bin" scripts/snapshot_qwen_ablation.py \
  --master data/annotations/xcl/ablation_2500/passes.jsonl --out "$subset_dir" \
  --seconds 100 500 1000 4000

mkdir -p "$artifact_dir"
checkpoints=()
for seconds in 100 500 1000 4000; do
  for variant in direct reasoning self_review shifted_review full; do
    checkpoint="$artifact_dir/$variant-s$seconds.pt"
    "$python_bin" scripts/train.py \
      --annotations "$subset_dir/${seconds}s/$variant.jsonl" \
      --shard-dir data/xcl/shards --backbone georgeven/songmae-large-32x1 \
      --train-all --out "$checkpoint"
    checkpoints+=(--checkpoint "$checkpoint")
  done
done

"$python_bin" scripts/evaluate_2d.py --dataset powdermill --root "$birdcode_root" \
  "${checkpoints[@]}" --out "$result_dir"
"$python_bin" scripts/plot_qwen_ablation_seconds.py \
  "$result_dir/iou_powdermill_77.json" --out "$result_dir/qwen_ablation_seconds"
