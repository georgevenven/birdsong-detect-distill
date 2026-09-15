#!/usr/bin/env bash
# Short temporal tasks run concurrently with the disjoint WABAD/Hawaii files.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1
model=$1
results=results/competitors_external_2026-09-08
model_args=()
case "$model" in
  birdbox) ;;
  qwen_yolo) model_args=(--checkpoint artifacts/baselines_matched_2026-09-07/yolo11n-self_review-10000s.pt) ;;
  *) exit 2 ;;
esac
for dataset in xcsl nips4bplus; do
  root=/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw
  if [[ "$dataset" == nips4bplus ]]; then root=data/nips4bplus; fi
  pids=()
  for i in 0 1; do
    (.venv/bin/python -u scripts/evaluate_baselines.py --model "$model" --dataset "$dataset" --root "$root" \
      --manifest results/baselines_matched_2026-09-07/manifest.json \
      --calibration "results/baselines_matched_2026-09-07/powdermill_final/$model.json" \
      --cache "artifacts/competitors_external_2026-09-08/$model" \
      --out "$results/parts/${model}_${dataset}_${i}.json" --shards 2 --shard-index "$i" "${model_args[@]}" \
      2>&1 | tee "$results/logs/${model}_${dataset}_${i}.log") &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if ((status)); then exit "$status"; fi
  .venv/bin/python scripts/merge_external_shards.py --model "$model" --dataset "$dataset" --shards 2
done
