#!/usr/bin/env bash
set -euo pipefail

python_bin=${PYTHON_BIN:-.venv/bin/python}
birdcode_python=${BIRDCODE_PYTHON:-.venv-birdcode/bin/python}
root=${BIRDCODE_ROOT:-/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw}
manifest=${BENCHMARK_MANIFEST:-results/baselines_matched_2026-09-07/manifest.json}
results=${BENCHMARK_RESULTS:-results/baselines_matched_2026-09-07/powdermill_final}
cache=${BENCHMARK_CACHE:-artifacts/baselines_matched_2026-09-07/paper}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HF_HUB_OFFLINE=1
mkdir -p "$results/logs"
event_args=()
if [[ ${EVENT_METRICS:-0} == 1 ]]; then event_args=(--event-metrics); fi

for model in birdbox qwen_yolo songmae birdcode; do
  runner=$python_bin
  model_cache=$cache/$model
  model_args=()
  if [[ "$model" == birdbox || "$model" == qwen_yolo ]]; then
    model_cache=$model_cache/confidence_1e-5
  fi
  if [[ "$model" == qwen_yolo ]]; then
    model_args=(--checkpoint artifacts/baselines_matched_2026-09-07/yolo11n-self_review-10000s.pt)
  elif [[ "$model" == songmae ]]; then
    model_args=(--checkpoint artifacts/qwen_ablation_ap_10000/self_review-s10000.pt
      --backbone-revision f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e)
  elif [[ "$model" == birdcode ]]; then
    runner=$birdcode_python
    model_args=(--batch-size 4)
  fi
  if [[ -f "$results/$model.json" ]]; then
    "$runner" -c 'import hashlib,json,pathlib,sys; d=json.load(open(sys.argv[1])); assert not d["diagnostic_only"] and d["event_metrics"] == bool(int(sys.argv[2])) and d["manifest_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest() and d["dataset"] == "powdermill" and d["inference"]["model"] == sys.argv[4] and (sys.argv[4] not in ("birdbox","qwen_yolo") or d["inference"]["confidence_floor"] == .00001), "existing report has different settings; choose BENCHMARK_RESULTS"' \
      "$results/$model.json" "${EVENT_METRICS:-0}" "$manifest" "$model"
    continue
  fi
  "$runner" -u scripts/evaluate_baselines.py --model "$model" --dataset powdermill --root "$root" \
    --manifest "$manifest" --cache "$model_cache" --out "$results/$model.json" \
    "${model_args[@]}" "${event_args[@]}" 2>&1 | tee "$results/logs/$model.log"
done
