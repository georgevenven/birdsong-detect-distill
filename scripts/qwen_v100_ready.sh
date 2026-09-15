#!/usr/bin/env bash
# Runs on the V100 host in tmux, after the independent build/download start.
set -euo pipefail
worker_root=/media/george/DATA/birdsong-powdermill-worker-20260913
deadline=$((SECONDS + 43200))
while (( SECONDS < deadline )); do
  if [[ -x "$worker_root/build-sm70/bin/llama-server" && -s "$worker_root/provenance/weights-verified.txt" ]]; then
    if ! tmux has-session -t qwen-v100-build-20260913 2>/dev/null; then
      "$worker_root/build-sm70/bin/llama-server" --version > "$worker_root/provenance/build-version.txt" 2>&1
      exec bash "$worker_root/qwen_v100_server.sh"
    fi
  fi
  sleep 10
done
echo 'Build/download readiness deadline exceeded.' >&2
exit 1
