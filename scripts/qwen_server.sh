#!/usr/bin/env bash
set -euo pipefail

model_dir=${MODEL_DIR:-models/Qwen3.8-27B-GGUF}
exec llama-server \
  --model "$model_dir/Qwen3.8-27B-Q8_0.gguf" \
  --host 0.0.0.0 --port "${PORT:-8080}" \
  --ctx-size "${CTX_SIZE:-16384}" --parallel "${PARALLEL:-8}"
