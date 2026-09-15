#!/usr/bin/env bash
# Native competitors only. Existing Powdermill calibration is immutable.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1
model=$1
results=results/competitors_external_2026-09-08
cache=artifacts/competitors_external_2026-09-08/$model
calibration=results/baselines_matched_2026-09-07/powdermill_final/$model.json
runner=.venv/bin/python
shards=${COMPETITOR_SHARDS:-4}
model_args=()
datasets=(wabad hawaii xcsl nips4bplus)
case "$model" in
  birdbox) ;;
  qwen_yolo) model_args=(--checkpoint artifacts/baselines_matched_2026-09-07/yolo11n-self_review-10000s.pt) ;;
  birdcode) runner=.venv-birdcode/bin/python; model_args=(--batch-size 4); datasets=(xcsl nips4bplus); shards=1 ;;
  *) exit 2 ;;
esac
mkdir -p "$results/parts"
for dataset in "${datasets[@]}"; do
  root=/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw
  if [[ "$dataset" == hawaii ]]; then root=data/hawaii/zenodo; fi
  if [[ "$dataset" == nips4bplus ]]; then root=data/nips4bplus; fi
  if [[ -f "$results/${model}_${dataset}.json" ]]; then
    "$runner" -c 'import json,sys; r=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); assert not r["diagnostic_only"] and r["dataset"]==sys.argv[3] and r["inference"]==c["inference"] and r["manifest_sha256"]==c["manifest_sha256"]' \
      "$results/${model}_${dataset}.json" "$calibration" "$dataset"
    continue
  fi
  pids=()
  for ((i=0; i<shards; i++)); do
    part="$results/parts/${model}_${dataset}_${i}.json"
    if [[ -f "$part" ]]; then continue; fi
    ("$runner" -u scripts/evaluate_baselines.py --model "$model" --dataset "$dataset" --root "$root" \
      --manifest results/baselines_matched_2026-09-07/manifest.json --calibration "$calibration" \
      --cache "$cache" --out "$part" --shards "$shards" --shard-index "$i" "${model_args[@]}" \
      2>&1 | tee "$results/logs/${model}_${dataset}_${i}.log") &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if ((status)); then exit "$status"; fi
  "$runner" scripts/merge_external_shards.py --model "$model" --dataset "$dataset" --shards "$shards"
done
