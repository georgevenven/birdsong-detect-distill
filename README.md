# birdsong-detect-distill

Distill reviewed Qwen spectrogram boxes into a dense bird-vocalization detector on a frozen SongMAE backbone.

This repository is standalone. SongMAE is loaded from the [Hugging Face model collection](https://huggingface.co/collections/georgeven/songmae-a-bioacoustic-encoder-for-birdsong-6a91eb9c42e5cde53962fbec); no SongMAE source checkout or local checkpoint path is required. The default detector uses [`georgeven/songmae-large-32x1`](https://huggingface.co/georgeven/songmae-large-32x1), whose 32×1 patches provide a 5 ms temporal grid.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Data

Reviewed Qwen annotations are committed under `data/annotations/xcl/`. Historical absolute shard paths were reduced to shard filenames; readers resolve them against `--shard-dir`.

The XCL spectrogram shards are intentionally not copied. Place them at `data/xcl/shards/`, or pass their location explicitly. The original 205 MB XCL annotation index is available as `XCL_train_annotations.json.gz`; its unpacked local copy is ignored because GitHub rejects files over 100 MB.

Foreground labels are `target_vocalization`, `uncertain_vocalization`, and `chorus`. The backbone remains frozen. One small Transformer layer attends over the complete 4×1000 SongMAE token grid and emits a 128×1000 pixel mask.

## Train

```bash
python scripts/train.py \
  --shard-dir data/xcl/shards \
  --backbone georgeven/songmae-large-32x1
```

The default training batch size is 16; validation uses 4 to avoid attention-memory fragmentation. Large 32×1 produces 4,000 tokens per five-second window.

The reproduced checkpoint trained on 1,655 windows and selected epoch 1 by validation loss. On the untouched half of the held-out recordings it reached 0.545 micro F1, 0.471 precision, and 0.645 recall against the Qwen teacher.

## Evaluate detection only

Download the WABAD, Powdermill, or XCSL archives into `data/birdcode/raw`, then run:

```bash
python scripts/evaluate_birdcode.py --dataset xcsl
python scripts/evaluate_birdcode.py --dataset powdermill
python scripts/evaluate_birdcode.py --dataset wabad
```

Evaluation collapses every species into one `bird` class. It follows BirdCODE's 101-threshold, 100 Hz, one-second event merging, Pascal-VOC AP, and IoU 0.2/0.5 protocol. WABAD is macro-averaged across 68 sites.
The paper used for the protocol is preserved at `docs/papers/BirdCODE.pdf`.

Evaluate the released BirdBox YOLO11n checkpoint without training:

```bash
python scripts/evaluate_birdbox.py --dataset xcsl
```

The script downloads the official checkpoint, verifies its SHA-256, and reproduces BirdBox's six-second 1024-pixel spectrogram pipeline and default inference settings.

## Distill into YOLO11n

Render the reviewed five-second Qwen windows, then fine-tune the released BirdBox checkpoint:

```bash
python scripts/prepare_yolo.py --shard-dir /path/to/XCL/shards
python scripts/train_yolo.py
python scripts/evaluate_qwen_yolo.py --dataset xcsl
```

Pass `--scratch` to train the identical YOLO11n architecture from random initialization. Images are split by recording, and `target_vocalization`, `uncertain_vocalization`, and `chorus` are collapsed into one `bird` class.

For a direct human-box comparison of YOLO and SongMAE in time-frequency space:

```bash
python scripts/evaluate_2d.py --dataset powdermill --root /path/to/birdcode/raw
```

WABAD can be divided by site with `--shards N --shard-index I`. Merge its JSON shards, and their adjacent raw-counter `.npz` files, with:

```bash
python scripts/merge_wabad.py results/reproduced/songmae_large_32x1_wabad_*of*.json
```

## Qwen annotation

`scripts/annotate_qwen.py` contains the current spectrogram-only workflow: primary annotation, mandatory self-review, an independent ±1-second shifted review, and conditional adjudication. Every event includes label confidence; unresolved simultaneous singers use `chorus`.

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-GGUF scripts/qwen_server.sh
python scripts/annotate_qwen.py --spec-dir data/xcl
python scripts/serve_annotations.py --shard-dir data/xcl/shards
```

## Migration notes

`results/reference/` preserves the previous SongMAE Large 32×4 and BirdCODE detection-only outputs. `results/visualizations/` contains held-out Large 32×1 predictions. `artifacts/legacy/` preserves the old head for provenance, but it is not paired with Large 32×1. New checkpoints record only a Hugging Face model ID and contain no path to another repository.
