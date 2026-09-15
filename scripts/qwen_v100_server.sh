#!/usr/bin/env bash
set -euo pipefail
worker_root=/media/george/DATA/birdsong-powdermill-worker-20260913
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_pids" ]]
exec "$worker_root/build-sm70/bin/llama-server" \
  --model "$worker_root/models/Qwen3.8-27B-Q8_0.gguf" \
  --mmproj "$worker_root/models/mmproj-F16.gguf" \
  --image-min-tokens 1024 --ctx-size 262144 --parallel 16 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 --ctx-checkpoints 0 \
  --n-gpu-layers 999 --device CUDA0,CUDA1,CUDA2 \
  --split-mode layer --tensor-split 1,1,1 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 --threads 8 --threads-batch 8 \
  --jinja --reasoning-effort high --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
  --host 127.0.0.1 --port 8001
