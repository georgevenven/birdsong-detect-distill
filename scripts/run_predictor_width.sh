#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
run=predictor_width_10000train_800val_2026-09-10
labels=data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
mkdir -p "$artifacts" "$results/logs"
.venv/bin/python scripts/summarize_predictor_width.py --prepare
.venv/bin/python - <<'PY'
import shutil, subprocess
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
if pids:
    raise SystemExit('A GPU process is active; refusing to interfere.')
if shutil.disk_usage('.').free < 3*1024**3:
    raise SystemExit('Need at least 3 GiB free for width experiment outputs.')
print('GPU ownership and disk checks passed. This runner never starts Qwen.',flush=True)
PY

lane() {
  local gpu=$1
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  local name size hidden revision
  for name in "$@"; do
    size=${name%_d*}
    hidden=${name#*_d}
    case "$size" in
      base) revision=d47970abb03ad01c575bd98ed21932283d00eaa0 ;;
      large) revision=f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e ;;
    esac
    if [[ ! -f "$artifacts/$name.pt" ]]; then
      .venv/bin/python -u scripts/train.py \
        --annotations "$labels/train/self_review.jsonl" --validation-annotations "$labels/validation/self_review.jsonl" \
        --backbone "georgeven/songmae-$size-32x1" --backbone-revision "$revision" \
        --epochs 5 --batch-size 16 --eval-batch-size 4 --accumulation 1 --seed 0 \
        --hidden "$hidden" --dropout 0.1 --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
        --log-every 25 --out "$artifacts/$name.pt" \
        2>&1 | tee -a "$results/logs/train_$name.log"
    fi
    .venv/bin/python -u scripts/evaluate_current_backbones.py --size "$size" --seconds 10000 \
      --seed 0 --training-epochs 5 --head-width "$hidden" --prediction-cache samples \
      2>&1 | tee -a "$results/logs/evaluate_$name.log"
  done
}

lane GPU-93189925-4747-5857-e01f-d44d77ad07a7 large_d768 & first=$!
lane GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556 base_d128 base_d384 & second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
if (( status != 0 )); then exit "$status"; fi
.venv/bin/python scripts/summarize_predictor_width.py
