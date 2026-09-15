#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
model=$1
results=results/current_external_2026-09-09
cache=artifacts/current_external_2026-09-09/$model
shards=${CURRENT_SHARDS:-4}
case "$model" in
  songmae) checkpoint=artifacts/stage_pipeline_10000train_800val_2026-09-09/self_review.pt ;;
  qwen_yolo) checkpoint=artifacts/yolo/self_review_10000train_800val_2026-09-09.pt ;;
  *) exit 2 ;;
esac
mkdir -p "$results/logs" "$results/parts"
calibration="$results/calibration/$model.json"
if [[ ! -f "$calibration" ]]; then
  .venv/bin/python -u scripts/evaluate_current_models.py --model "$model" --dataset powdermill \
    --root /home/george-vengrovski/Documents/SongMAE/data/birdcode/raw --manifest "$results/manifest.json" \
    --checkpoint "$checkpoint" --cache "$cache" --out "$calibration" \
    2>&1 | tee -a "$results/logs/${model}_calibration.log"
fi
for dataset in nips4bplus xcsl wabad hawaii; do
  if [[ -f "$results/${model}_${dataset}.json" ]]; then continue; fi
  root=/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw
  if [[ "$dataset" == hawaii ]]; then root=data/hawaii/zenodo; fi
  if [[ "$dataset" == nips4bplus ]]; then root=data/nips4bplus; fi
  pids=()
  for ((i=0; i<shards; i++)); do
    part="$results/parts/${model}_${dataset}_${i}.json"
    if [[ -f "$part" ]]; then continue; fi
    (.venv/bin/python -u scripts/evaluate_current_models.py --model "$model" --dataset "$dataset" \
      --root "$root" --manifest "$results/manifest.json" --checkpoint "$checkpoint" \
      --cache "$cache" --out "$part" --calibration "$calibration" --shards "$shards" --shard-index "$i" \
      2>&1 | tee -a "$results/logs/${model}_${dataset}_${i}.log") &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if ((failed)); then exit 1; fi
  .venv/bin/python scripts/merge_current_external.py --results "$results" --model "$model" --dataset "$dataset" --shards "$shards"
done
