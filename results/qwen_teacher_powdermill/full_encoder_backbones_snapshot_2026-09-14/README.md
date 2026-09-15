# Full-encoder Micro / Base / Large comparison

Frozen 2026-09-14 snapshot: 801 unique five-second XC windows, all with accepted reasoning and one self-review. Twins completed 407 reviews and V100 completed 394. New annotations arriving after this snapshot are excluded from every model in this comparison.

- Training: all 641 non-validation windows, **3,205 seconds**. This includes the earlier 2,000-second training subset.
- Validation: the unchanged 160 source-disjoint windows, **800 seconds**, used in the previous detector experiments. Known benchmark/validation exclusions from the 50k annotation queue are checked again.
- Models: SongMAE Micro, Base, Large; full detection encoder fine-tuned, including CNN and positional embeddings, with a bare linear head and sigmoid. No added attention layer or detector LayerNorm.
- Seed 0, five epochs, checkpoint selected by minimum XC validation hard-mask BCE. AdamW: encoder LR 1e-5, head LR 1e-3, weight decay 1e-4. FP32; microbatch 2, effective batch 16 (final batch may be smaller). No soft targets, confidence-weighted targets, TV, or active dropout.
- Same inputs and targets are hash-checked across sizes. Reset sampling RNG after architecture initialization so all three sizes see identical minibatch orders; verify epoch order hashes before writing the combined summary.
- Evaluate all 77 Powdermill five-minute segments across all four original recordings. Each held-out original recording uses a threshold calibrated on the other three. Retain the legacy Recording_1 report separately.
- Unchanged inference: native frontend, five-second windows / 2.5-second stride, maximum overlap aggregation, Gaussian probability smoothing with sigma (2 mel bins, 3 frames), one calibrated threshold. Pixel AP uses continuous smoothed scores. No morphological cleanup.

One seed and a shared learning-rate recipe: this is a development comparison, not a per-size optimization or an independent test. Earlier 2k and frozen-encoder results are preserved, not substituted as matched controls.

## Execution

Large runs on Twins GPU 0; Micro then Base run on GPU 1. Each model is trained and evaluated before its lane advances. V100 annotation remains running; this job neither starts nor stops Qwen.

Detached service: `birdsong-full-encoder-backbones-snapshot-20260914.service`. User lingering is enabled. Epoch-boundary resume checkpoints and per-segment evaluation caches permit restart after interruption; automatic failure restarts are bounded.

Run/restart: `.venv/bin/python scripts/run_full_encoder_backbones_snapshot.py` from the repository root with Twins GPUs free. The existing manifest is reused, never resampled.

Monitor `status.json`, `runs/{micro,base,large}/status.json`, each model's `.log`, or `journalctl --user -u birdsong-full-encoder-backbones-snapshot-20260914.service`.

Final metrics: `summary.json`; per-model `runs/{size}/loro.json`, `comparison.json`, `history.json`, and `checkpoint.json`. The combined summary is written only after all models finish and their sample-order hashes match.

Weights, optimizer state, and sampled probability caches live off the nearly full root disk at `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/full_encoder_backbones_snapshot_2026-09-14/`.
