#!/usr/bin/env bash
set -euo pipefail

python_bin=${PYTHON_BIN:-.venv/bin/python}
queue=${1:?Pass the frozen continuation tiles.jsonl}
trap 'exit 130' INT TERM

for attempt in 1 2 3; do
  echo "XC continuation pass $attempt/3: 16 workers, 16,384-token server slots"
  set +e
  "$python_bin" scripts/annotate_qwen.py --tiles-from "$queue" \
    --workers 16 --reasoning-budget 1024 --max-tokens 4096
  status=$?
  set -e
  if (( status == 0 )); then
    exit 0
  fi
  if (( status != 2 )); then
    exit "$status"
  fi
done
echo "Some XC windows still need inspection after three passes; stage checkpoints are preserved." >&2
exit 2
