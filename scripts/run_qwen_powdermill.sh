#!/usr/bin/env bash
set -euo pipefail

root=${BIRDCODE_ROOT:-/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw}
python_bin=${PYTHON_BIN:-.venv/bin/python}
output=data/annotations/powdermill/qwen_ablation

while true; do
  set +e
  "$python_bin" scripts/annotate_qwen_ablation.py \
    --spec-dir data/powdermill/qwen \
    --tiles-from data/powdermill/qwen/recordings.jsonl \
    --xcaj-audio "$root/xcaj/Audio.zip" \
    --out-dir "$output" --recordings 77 \
    --workers 32 --reasoning-budget 2048 --max-tokens 4096
  status=$?
  set -e
  if (( status == 0 )); then
    break
  fi
  if (( status != 2 )); then
    exit "$status"
  fi
done

systemctl --user stop songmae-qwen.service
"$python_bin" scripts/evaluate_qwen_teacher.py \
  --annotations "$output/passes.jsonl" --root "$root" \
  --out results/qwen_teacher_powdermill/scores.json
