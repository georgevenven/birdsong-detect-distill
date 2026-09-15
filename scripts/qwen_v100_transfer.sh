#!/usr/bin/env bash
# Runs detached on the viewing desktop; never deletes source files.
set -euo pipefail
worker_root=/media/george/DATA/birdsong-powdermill-worker-20260913
source_root=/media/george-vengrovski/disk2/birdsong-powdermill-worker-20260911
export RSYNC_RSH='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3'
rsync -a --partial --append-verify --info=progress2 \
  "$source_root/models/Qwen3.8-27B-Q8_0.gguf" \
  george@george-server:"$worker_root/models/"
rsync -a --partial --append-verify --info=progress2 \
  /home/george-vengrovski/models/Qwen3.8-27B-mmproj-F16.gguf \
  george@george-server:"$worker_root/models/mmproj-F16.gguf"
ssh -o BatchMode=yes george@george-server \
  "cd '$worker_root/models' && sha256sum -c '$worker_root/weights.sha256'"
