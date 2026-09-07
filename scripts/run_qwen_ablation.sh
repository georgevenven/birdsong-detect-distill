#!/usr/bin/env bash
set -euo pipefail

birdcode_root=${BIRDCODE_ROOT:-/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw}
python_bin=${PYTHON_BIN:-.venv/bin/python}
variant_dir=data/annotations/xcl/ablation_10
artifact_dir=artifacts/qwen_ablation_10

"$python_bin" scripts/audit_overlap.py \
  --xcl-val-index /home/george-vengrovski/Documents/SongMAE/data/XCL_val/shards/index.tsv \
  --birdcode-root "$birdcode_root"
"$python_bin" scripts/annotate_qwen_ablation.py --xcaj-audio "$birdcode_root/xcaj/Audio.zip" \
  --recording XC172822 --recording XC45355 --recording XC594278 --recording XC633294 \
  --recording XC686801 --recording XC728036 --recording XC732921 --recording XC388500 \
  --recording XC742088 --recording XC471480

if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null; then
  echo "Qwen annotation is complete. Stop the Qwen server and rerun this command for training." >&2
  exit 3
fi

mkdir -p "$artifact_dir"
for variant in direct reasoning self_review shifted_review full; do
  "$python_bin" scripts/train.py \
    --annotations "$variant_dir/$variant.jsonl" \
    --shard-dir data/xcl/shards \
    --backbone georgeven/songmae-large-32x1 \
    --train-all \
    --out "$artifact_dir/$variant.pt"
done

checkpoints=()
for variant in direct reasoning self_review shifted_review full; do
  checkpoints+=(--checkpoint "$artifact_dir/$variant.pt")
done
"$python_bin" scripts/evaluate_2d.py --dataset powdermill --root "$birdcode_root" \
  "${checkpoints[@]}" --out results/qwen_ablation_10
