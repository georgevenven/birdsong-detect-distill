#!/usr/bin/env bash
# Authorized removal of inventoried, obsolete Qwen3.6 model weights only.
set -euo pipefail
worker_root=/media/george/DATA/birdsong-powdermill-worker-20260913
[[ $(hostname) == george-server ]]
[[ $(systemctl is-active llama-qwen.service || true) == inactive ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
mkdir -p "$worker_root/provenance" "$worker_root/models" "$worker_root/logs"
cp -n /etc/systemd/system/llama-qwen.service "$worker_root/provenance/llama-qwen.old.service"
old_weights=(
  /home/george/models/qwen3.6-27b-mtp-q4-k-m/Qwen3.6-27B-Q4_K_M.gguf
  /home/george/models/qwen3.6-27b-mtp-q5-k-m/Qwen3.6-27B-Q5_K_M.gguf
  /home/george/models/qwen3.6-35b-a3b-mtp-q5-k-m/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
  /media/george/DATA/model-archive/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf
  /media/george/DATA/model-archive/qwen3.6-27b-mtp/Qwen3.6-27B-Q5_K_M.gguf
  /media/george/DATA/model-archive/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
  /media/george/DATA/model-archive/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  /media/george/DATA/model-archive/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
)
for weight in "${old_weights[@]}"; do
  [[ -f "$weight" && ! -L "$weight" && $(realpath "$weight") == "$weight" ]]
  stat --format='%s %n' "$weight"
done
rm -v -- "${old_weights[@]}"
df -h / /media/george/DATA
