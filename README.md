# birdsong-detect-distill

Distill reviewed Qwen spectrogram boxes into a dense bird-vocalization detector using SongMAE.

This repository is standalone. Pretrained SongMAE is loaded from the [Hugging Face model collection](https://huggingface.co/collections/georgeven/songmae-a-bioacoustic-encoder-for-birdsong-6a91eb9c42e5cde53962fbec); no SongMAE source checkout is required. The default detector uses [`georgeven/songmae-large-32x1`](https://huggingface.co/georgeven/songmae-large-32x1), whose 32×1 patches provide a 5 ms temporal grid. Detection also requires the trained detector checkpoint.

## Current default

As of 2026-09-14, use **fully fine-tuned SongMAE-Large + a bare linear head and sigmoid** for future detector work, followed by Gaussian probability smoothing and one calibrated threshold. This is a qualitative model-selection decision, not a claim of best benchmark AP. See the [selected checkpoint, training recipe, and inference policy](docs/default-detector.md).

Historical checkpoints, experiment scripts, and benchmark results remain unchanged; the frozen-backbone training instructions below reproduce the earlier baseline.

The [2026-09-15 experiment snapshot](docs/experiments-2026-09-15.md) records the
completed 15k scaling/backbone runs, external results, and the current annotation
target. The paper evaluation queue is paused; Twins and V100 annotate toward
27,500 total seconds for a future 25,000/2,500-second train/validation split.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Data

Historical reviewed Qwen annotations are committed under `data/annotations/xcl/`.
New live annotation runs, checkpoints, full provenance/caches, and generated figures
stay local and must be transferred separately. Compact experiment summaries and
plotting code are versioned. Historical absolute shard paths were reduced to shard
filenames; readers resolve them against `--shard-dir`.

The XCL spectrogram shards are intentionally not copied. Place them at `data/xcl/shards/`, or pass their location explicitly. The original 205 MB XCL annotation index is available as `XCL_train_annotations.json.gz`; its unpacked local copy is ignored because GitHub rejects files over 100 MB.

[Hawaii data and provenance](docs/hawaii.md) documents the downloaded original Zenodo FLAC release, its time–frequency labels, and source-annotation caveats for later evaluation.

Historical foreground labels are `target_vocalization`, `uncertain_vocalization`, and `chorus`. In the frozen-backbone baseline, one small Transformer layer attends over the complete 4×1000 SongMAE token grid and emits a 128×1000 pixel mask.

## Historical frozen-backbone training

```bash
python scripts/train.py \
  --shard-dir data/xcl/shards \
  --backbone georgeven/songmae-large-32x1
```

The default training batch size is 16; validation uses 4 to avoid attention-memory fragmentation. Large 32×1 produces 4,000 tokens per five-second window.

The historical reproduced checkpoint trained on 1,655 windows. The earlier matched baseline comparison uses all 2,103 windows / 10,000 seconds of self-review labels for both SongMAE and YOLO.

The [2026-09-09 four-stage ablation](results/qwen_teacher_powdermill/stage_pipeline_10000train_800val_2026-09-09/README.md)
uses exactly 10,000 s for training and 800 s from separate XC recordings for checkpoint selection.
It uses hard targets and plain BCE (`--no-target-smoothing --tv-weight 0`), then probability smoothing
and a single Powdermill-calibrated threshold. Run `bash scripts/run_stage_pipeline.sh`; historical results remain separate.

## Matched baseline evaluation

See [baseline inputs, splits, metrics and environment setup](docs/baselines.md). The shared evaluator preserves each model's native input and scores identical intervals. The frozen manifest reserves original Powdermill Recordings 2–4 for thresholds and the teacher-covered intervals of Recording_1 for reporting.

```bash
BIRDCODE_ROOT=/path/to/birdcode/raw bash scripts/run_matched_baselines.sh
python scripts/evaluate_baselines.py --help
```

Primary metrics are exact foreground area AP and development-calibrated IoU, not bounding-box matching. Report both full-band and common-band (500–12,000 Hz) masks; BirdCODE supports temporal metrics only. Native-event scoring is optional. These detection-only metrics are not the papers' species-aware or IoMin metrics. Powdermill is development data, including for BirdCODE.

External runs require `--calibration` from the same model's complete Powdermill report. The manifest excludes known XCL training/validation overlap from XC-AJ; WABAD includes every recording from its 68 annotated sites; NIPS4Bplus excludes non-bird classes from positive labels. Historical results are not interchangeable with these reports.

## Distill into YOLO11n

Render the reviewed five-second Qwen windows, then fine-tune the released BirdBox checkpoint:

```bash
python scripts/prepare_yolo.py \
  --annotations data/annotations/xcl/ablation_2500/ap_10000/10000s/self_review.jsonl \
  --shard-dir data/xcl/shards --train-all --out data/yolo/self_review_10000s
python scripts/train_yolo.py --data data/yolo/self_review_10000s/dataset.yaml \
  --out artifacts/yolo/self_review_10000s --checkpoint artifacts/yolo/self_review_10000s.pt \
  --epochs 50 --batch-size 16 --device 0
```

Set `CUDA_VISIBLE_DEVICES` to the available GPU before training. With `--train-all`, the complete label budget trains the model for a fixed epoch count and the final checkpoint is used; training diagnostics are not held-out validation. Pass `--scratch` for random initialization. Without `--train-all`, preparation retains the older recording-level validation split.

`evaluate_2d.py` remains a preliminary Powdermill development tool for older ablation scripts; use the shared evaluator for paper comparisons. The former misleadingly named SongMAE-only `evaluate_birdcode.py` is preserved as `evaluate_songmae_temporal_legacy.py`; the current BirdCODE entry point runs the actual released model.

## Qwen annotation

The current XC workflow uses the full-Powdermill prompt: normalized coordinate
axes, reasoning, then **one numbered self-review**, with each stage saved separately.
All accepted events use `bird_vocalization`; confidence is retained. Sampling is
species/geography-balanced, one five-second window per source recording, with
known evaluation and reserved-validation IDs excluded.

`scripts/prepare_qwen_xc_50k.py` prepares the immutable queue;
`scripts/run_qwen_xc_50k.py` runs its shared-lock Twins/V100 clients. Each backend
uses 16 concurrent slots with 16,384 context tokens each. The original queue has
50,000 seconds, but `scripts/pause_xc_at_budget.py` now pauses both clients after
27,500 validated seconds, draining in-flight requests. See the
[budget-watcher handoff](results/qwen_xc_budget_27500_2026-09-15/README.md).

### Historical annotation pipelines

`scripts/annotate_qwen.py` preserves the earlier spectrogram-only workflow: primary
annotation, mandatory self-review, an independent ±1-second shifted review, and
conditional adjudication. Its labels include confidence and a separate `chorus` class.

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-GGUF scripts/qwen_server.sh
python scripts/annotate_qwen.py --spec-dir data/xcl
python scripts/serve_annotations.py --shard-dir data/xcl/shards
```

The server defaults to 16 slots sharing 262,144 context tokens (16,384 per slot).
`scripts/run_qwen_powdermill.sh` uses 16 workers; its reasoning and output limits remain 2,048 and 4,096 tokens.
Zero-budget calls explicitly disable thinking. Historical `direct` labels used a zero budget with thinking enabled;
they remain cached and must not be described as a verified no-reasoning baseline.

The original XC run resumes with `scripts/run_xc_continuation.sh data/annotations/xcl/continuation_2026-09-09/tiles.jsonl`.
Its frozen queue retains the original windows and adds 816 fresh five-second windows, excluding XC-AJ and the reserved
validation IDs from new sampling. It targets 23,208.95 s total (22,133.995 s outside validation). The append-only original
master retains all review passes; this is not a new student training split. The client uses 16 workers, its original
1,024-token reasoning budget, and a larger 4,096-token output cap. Each stage is checkpointed; SIGINT/SIGTERM drains active
windows, and retries stop after three passes. Run metadata records the current prompts and token settings separately from
historical labels. `birdsong-qwen-xc-continuation.service` runs annotation only, without automatic training/evaluation.

The cumulative ten-recording prompt ablation measures direct annotation, private reasoning, self-review, shifted review, and
conditional adjudication on identical XCL recordings. It excludes every XCAJ recording ID, trains one Large 32×1 head per
condition, and evaluates recording-mean temporal and 2D IoU only on Powdermill:

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-GGUF scripts/qwen_server.sh
BIRDCODE_ROOT=/path/to/birdcode/raw scripts/run_qwen_ablation.sh
```

After annotation completes, stop the Qwen server and rerun `scripts/run_qwen_ablation.sh`; completed annotations are skipped and
the freed GPUs are used for training and Powdermill evaluation.

## Migration notes

`results/reference/` preserves the previous SongMAE Large 32×4 and BirdCODE detection-only outputs. `results/visualizations/` contains held-out Large 32×1 predictions. `artifacts/legacy/` preserves the old head for provenance, but it is not paired with Large 32×1. New checkpoints record only a Hugging Face model ID and contain no path to another repository.
