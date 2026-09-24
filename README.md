# birdsong-detect-distill

Time-frequency bird vocalization detection by distillation: a vision-language model
(Qwen3.8-27B) boxes vocalizations on spectrograms, and those boxes train a SongMAE encoder
with a linear detection head that predicts a per-pixel vocalization probability.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r env/requirements-main.txt --extra-index-url https://download.pytorch.org/whl/cu118
.venv/bin/pip install -e . --no-deps
```

`env/requirements-main.txt` pins the versions used for the paper (Python 3.12); the YOLO11
teacher-label runs used `env/requirements-yolo-hexeberg.txt` (Ultralytics 8.3.109).

Scripts import each other, so run them with `PYTHONPATH=src:scripts` from the repository root.

## Detect vocalizations

```bash
PYTHONPATH=src:scripts python scripts/predict.py recording.wav \
    --checkpoint large_25000_s0.pt --out predictions/
```

Writes `predictions/recording.npz` (probability and mask, 128 mel bins × 5 ms frames) and
`predictions/recording.csv` (one row per detected region: start/end seconds, low/high Hz,
peak probability). Inference is 32 kHz, 128 mel bins (20–16,000 Hz), 5-s windows with
2.5-s overlap, Gaussian smoothing σ = (2 bins, 3 frames), and `--threshold` (default 0.04,
calibrated on Powdermill for SongMAE-Large). The backbone is fetched from Hugging Face
(`georgeven/songmae-{micro,base,large}-32x1`) unless `--backbone` names a local copy.

### Example

![Black Wheatear, XC839867: spectrogram and detected regions](docs/example-XC839867-black-wheatear.png)

Black Wheatear, [XC839867](https://xeno-canto.org/839867), not used in training: mel spectrogram
(top) and regions detected by SongMAE-Large (bottom, threshold 0.04). The detectors are on
Hugging Face as [georgeven/songmae-{micro,base,large}-32x1-bird-detector](https://huggingface.co/georgeven/songmae-large-32x1-bird-detector).
Recording by Esperanza Poveda, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
the image is an adaptation under the same license.

## Pipeline

| Stage | Scripts |
|---|---|
| Serve the teacher | `ops/qwen/qwen38_27b_server.sh` (llama.cpp, Qwen3.8-27B Q8_0 + mmproj), `ops/qwen/qwen_v100_*.sh` (second host) |
| Teacher prompts and parsing | `src/birdsong_detect_distill/qwen.py` |
| Teacher ablation on Powdermill | `prepare_powdermill_qwen.py`, `run_qwen_prompt_study.py`, `run_qwen_distributed.py`, `run_qwen_full_powdermill.py`, `summarize_powdermill_loro.py` |
| Teacher labels on Xeno-Canto | `prepare_qwen_xc_50k.py`, `run_qwen_xc_50k.py --role twins\|v100`, `pause_xc_at_budget.py` |
| Training windows | `export_xcl_windows.py` (label windows from BirdSet XCL shards) |
| Train and evaluate students | `paper_25k_suite.py` (driver) → `train.py`, `paper_25k_evaluate.py`, `paper_suite_evaluate.py`, `evaluate_2d.py`, `evaluate_songmae_smoothing.py`; data-budget sweep `run_scaling.py`, `run_scaling_seeds.py`, `run_v100_scaling_seeds.py`, `run_v100_large_checkpointed_seeds.py` |
| Baselines | `evaluate_baselines.py`, `baseline_models.py`, `birdbox.py` (released BirdBox YOLO11, BirdCODE) |
| YOLO11 on teacher labels | `run_hexeberg_yolo.py`, `hexeberg_suite.py`, `prepare_hexeberg_audio.py`, `evaluate_hexeberg_twins.py`, `queue_hexeberg_*.py`, `hexeberg_*.py`, `launch_hexeberg_*.sh` |

Library: `data.py` (spectrogram shards, pixel targets), `model.py` and
`full_encoder_linear.py` (backbone loading, linear head), `benchmark_data.py` and
`evaluation.py` (Powdermill, WABAD, XC-AJ, NIPS4Bplus, Hawaii loaders),
`benchmark_metrics.py` (pixel/frame AP and IoU).

## Models and data

Trained detectors, pretrained backbones, baselines, the Qwen GGUF, teacher labels, and
training spectrograms are not in git. On Twins they are in `artifacts/` and `data/` of
`/mnt/birdconv/songmae_perch2/vlm-teacher-bird-detection` (`~/Documents/vlm-teacher-bird-detection`):
`artifacts/models/songmae_students/large_25000_s{0,1,2}.pt` are the paper's SongMAE-Large
detectors; `artifacts/models/models.json` lists every model with its SHA-256 and origin.

Scripts are kept as they ran; some still name original machine paths
(`~/Documents/SongMAE/data/birdcode/raw` for benchmark audio, `/media/george/DATA` on the
second host). The pre-cleanup development history is on the `dev` branch.
