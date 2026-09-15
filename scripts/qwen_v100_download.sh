#!/usr/bin/env bash
# Exact revision and SHA-256 match the existing Powdermill backends.
set -euo pipefail
worker_root=/media/george/DATA/birdsong-powdermill-worker-20260913
source_url=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe
cd "$worker_root/models"
for weight in Qwen3.8-27B-Q8_0.gguf mmproj-F16.gguf; do
  curl --http1.1 -fL --retry 10 --retry-all-errors --retry-delay 15 --connect-timeout 30 \
    --speed-limit 10000 --speed-time 120 -C - \
    "$source_url/$weight" -o "$weight.download"
  expected=$(awk -v file="$weight" '$2 == file {print $1}' "$worker_root/weights.sha256")
  [[ $(sha256sum "$weight.download" | cut -d ' ' -f 1) == "$expected" ]]
  mv -- "$weight.download" "$weight"
done
sha256sum -c "$worker_root/weights.sha256" > "$worker_root/provenance/weights-verification.tmp"
mv "$worker_root/provenance/weights-verification.tmp" "$worker_root/provenance/weights-verified.txt"
