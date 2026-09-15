# Training-loss ablation: SongMAE-Large, 10k seconds

All four models were retrained using the same self-reviewed Qwen labels: 2,103 windows from 266 XC recordings. Same seed (0), initial adapter, minibatch order, frozen backbone revision, AdamW and final checkpoint after three epochs. No training/validation split was introduced.

Inference is fixed: probability-space Gaussian smoothing (σ = 2 mel bins, 3 frames), threshold 0.08; no other mask cleanup. The threshold comes from the previous control's calibration on original Powdermill Recordings 2–4; it was not reselected.

Identical 8,235 seconds across 30 segments of original Recording_1. AP is exact pixel AP of continuous smoothed scores, averaged over 29 positive segments; IoU is averaged over all 30 (empty union = 1). Precision/recall pool pixel counts.

| Training loss | Mask AP ↑ | 2D IoU ↑ | Precision | Recall |
|---|---:|---:|---:|---:|
| Soft targets + TV (control) | 0.768 | 0.516 | 0.619 | 0.720 |
| Soft targets, no TV | 0.768 | 0.516 | 0.619 | 0.720 |
| Hard targets + TV | 0.797 | 0.542 | 0.697 | 0.664 |
| Hard targets, no TV (BCE only) | 0.797 | 0.542 | 0.697 | 0.664 |

These are single-seed development results, not significance estimates or unseen-test performance. Removing target smoothing can change confidence calibration, so fixed-threshold IoU and threshold-free AP answer different questions.

Recommendation from this ablation: use hard foreground masks with plain BCE. Relative to the rerun control, removing both additions improves mask AP by 0.02930 and IoU by 0.02535. Precision rises by 0.07755 and recall falls by 0.05599 at the unchanged threshold. Removing TV alone changes AP by only about 0.00003 under either target type; there is no meaningful benefit from TV at the tested weight. Training defaults and previous checkpoints have not been replaced.

The old checkpoint reproduced its saved raw predictions bit-for-bit on all 30 segments using the current inference path. Rerun control head identical to the old head: False; AP difference: -0.00000049; IoU difference: -0.00000022.

`protocol.json` records the frozen design; `comparison.json` contains full precision results and paired-training checks; `part0.json`/`part1.json` include source/code hashes. Checkpoints and full-resolution probability caches are under `artifacts/training_loss_ablation_2026-09-08`. The annotation pause snapshot is preserved there too.

An independent direct-threshold audit reproduced TP/FP/FN for all 120 condition/segment masks and verified every prediction-cache checksum. Qwen annotation resumed at 13:26 PDT on September 8; the prior annotation-file prefix and all 1,267 stage-checkpoint files were preserved. Two resumed windows logged parse errors and remain eligible for the existing completion/retry loop. See `audit.json`.

To reproduce in fresh output directories, run both lanes of `scripts/run_training_loss_ablation.sh` (arguments `0` and `1`), then run `scripts/evaluate_training_loss_ablation.py --shard 0` and `--shard 1` on separate GPUs with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1`. Finally run `scripts/summarize_training_loss_ablation.py`. Update the dated output paths first: the scripts refuse to overwrite historical checkpoints or evaluations.
