#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for model in songmae yolo; do
  unit="birdsong-${model}-external-20260909.service"
  while systemctl --user is-active --quiet "$unit"; do sleep 30; done
done
for model in songmae qwen_yolo; do
  for dataset in nips4bplus xcsl wabad hawaii; do
    [[ -f "results/current_external_2026-09-09/${model}_${dataset}.json" ]] || exit 1
  done
done
exec .venv/bin/python scripts/summarize_current_external.py --results results/current_external_2026-09-09
