#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=GPU-93189925-4747-5857-e01f-d44d77ad07a7
run=confidence_targets_10000train_800val_2026-09-10
labels=data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09
artifacts="artifacts/$run"
results="results/qwen_teacher_powdermill/$run"
mkdir -p "$artifacts" "$results/logs"
.venv/bin/python scripts/summarize_uncertain_ablation.py --confidence-targets --prepare
.venv/bin/python - <<'PY'
import os, shutil, subprocess
from pathlib import Path
active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
if os.environ['CUDA_VISIBLE_DEVICES'] in active:
    raise SystemExit('Selected GPU is occupied; refusing to interfere.')
if shutil.disk_usage('.').free < 3 * 1024**3:
    raise SystemExit('Need at least 3 GiB free.')
if not (Path.home() / '.config/birdsong-detect-distill/qwen.paused').exists():
    raise SystemExit('Expected Qwen pause marker is absent.')
print('GPU and disk checks passed; this runner never starts Qwen.', flush=True)
PY
if [[ ! -f "$artifacts/large_d384_confidence.pt" ]]; then
  .venv/bin/python -u scripts/train.py \
    --annotations "$labels/train/self_review.jsonl" --validation-annotations "$labels/validation/self_review.jsonl" \
    --backbone georgeven/songmae-large-32x1 --backbone-revision f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e \
    --epochs 5 --batch-size 16 --eval-batch-size 4 --accumulation 1 --seed 0 \
    --hidden 384 --head-layers 1 --confidence-targets --dropout 0.1 --weight-decay 0.0001 --tv-weight 0 --no-target-smoothing \
    --log-every 25 --out "$artifacts/large_d384_confidence.pt" \
    2>&1 | tee -a "$results/logs/train_large_d384_confidence.log"
fi
.venv/bin/python -u scripts/evaluate_current_backbones.py --size large --seconds 10000 --seed 0 \
  --training-epochs 5 --head-width 384 --head-layers 1 --confidence-targets --fixed-large-width --prediction-cache samples \
  2>&1 | tee -a "$results/logs/evaluate_large_d384_confidence.log"
.venv/bin/python scripts/summarize_uncertain_ablation.py --confidence-targets
