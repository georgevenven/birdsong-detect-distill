#!/usr/bin/env bash
set -euo pipefail

birdcode_root=${BIRDCODE_ROOT:-/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw}
python_bin=${PYTHON_BIN:-.venv/bin/python}
variant_dir=data/annotations/xcl/ablation_2500
artifact_dir=artifacts/qwen_ablation_2500
result_dir=results/qwen_ablation_2500

"$python_bin" scripts/audit_overlap.py \
  --xcl-val-index /home/george-vengrovski/Documents/SongMAE/data/XCL_val/shards/index.tsv \
  --birdcode-root "$birdcode_root"

"$python_bin" scripts/annotate_qwen_ablation.py \
  --xcaj-audio "$birdcode_root/xcaj/Audio.zip" \
  --out-dir "$variant_dir" --recordings 275 --max-tiles 2500 \
  --workers 32 --reasoning-budget 2048 --max-tokens 4096

systemctl --user stop songmae-qwen.service
mkdir -p "$artifact_dir"
for variant in direct reasoning self_review shifted_review full; do
  "$python_bin" scripts/train.py \
    --annotations "$variant_dir/$variant.jsonl" \
    --shard-dir data/xcl/shards \
    --backbone georgeven/songmae-large-32x1 \
    --train-all --out "$artifact_dir/$variant.pt"
done

checkpoints=()
for variant in direct reasoning self_review shifted_review full; do
  checkpoints+=(--checkpoint "$artifact_dir/$variant.pt")
done
"$python_bin" scripts/evaluate_2d.py --dataset powdermill --root "$birdcode_root" \
  "${checkpoints[@]}" --out "$result_dir"
