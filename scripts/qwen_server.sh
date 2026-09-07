#!/usr/bin/env bash
set -euo pipefail

model_dir=${MODEL_DIR:-models/Qwen3.8-27B-GGUF}
exec llama-server \
  --model "$model_dir/Qwen3.8-27B-Q8_0.gguf" \
  --mmproj "$model_dir/mmproj-F16.gguf" \
  --image-min-tokens 1024 \
  --ctx-size "${CTX_SIZE:-262144}" \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-gpu-layers 999 --device CUDA0,CUDA1 \
  --split-mode layer --tensor-split 1,1 \
  --flash-attn on --batch-size 2048 --ubatch-size 512 \
  --parallel "${PARALLEL:-32}" --jinja \
  --reasoning-effort high --reasoning-preserve \
  --host 0.0.0.0 --port "${PORT:-8080}" \
  --temp .6 --top-p .95 --top-k 20 --min-p 0
