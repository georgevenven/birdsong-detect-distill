#!/usr/bin/env bash
set -euo pipefail
helper_root=/media/george-vengrovski/disk2/birdsong-powdermill-worker-20260911
exec "$helper_root/build/bin/llama-server" \
  --model "$helper_root/models/Qwen3.8-27B-Q8_0.gguf" \
  --mmproj /home/george-vengrovski/models/Qwen3.8-27B-mmproj-F16.gguf \
  --image-min-tokens 1024 --ctx-size 131072 --parallel 8 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-gpu-layers auto --device CUDA0 --split-mode layer \
  --fit on --fit-target 2000 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 --threads 8 --threads-batch 8 \
  --jinja --reasoning-effort high --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
  --host 127.0.0.1 --port 8081
