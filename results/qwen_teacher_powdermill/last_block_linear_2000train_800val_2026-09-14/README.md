# Final encoder-block fine-tuning

Single seed-0 pilot against the existing frozen SongMAE-Large + bare linear control.

- Train exactly the same 2,000 s of XC; select the best of five epochs by BCE on the same source-disjoint 800 s XC validation set.
- Train `songmae.encoder.layers.11` (7,087,872 parameters) and `Linear(768,32)` (24,608). All other backbone tensors stay frozen. The backbone remains in evaluation mode to preserve the control's disabled dropout.
- AdamW: head LR 1e-3, last-block LR 1e-5, weight decay 1e-4, batch 16. Hard-mask BCE, no soft targets or TV. No trained detector warm-start; identical initial head and checked minibatch order/CPU RNG.
- Save only the trained block and head; inference must use the specialized `last_block_linear.load_heads` loader, not the frozen-backbone loader.
- Same Powdermill inference and probability smoothing: sigma 2 mel bins / 3 frames (15 ms), then one threshold. Pixel AP uses continuous smoothed probabilities.
- Score all 77 five-minute Powdermill segments. For each original recording, calibrate its threshold on the other three. Also retain the legacy Recording_1-only result. No external benchmarks.

Completed successfully; both models selected epoch 4 by XC validation loss.

| Seed-0 model | Pixel AP | Mean 2D IoU |
|---|---:|---:|
| Frozen SongMAE-Large + linear | 0.761469 | 0.508836 |
| Last encoder block unfrozen + linear | 0.766917 | 0.513097 |
| Unfrozen minus frozen | +0.005448 | +0.004261 |

These are full-Powdermill leave-one-recording-out development scores, with segment-macro aggregation. AP excludes the one segment with no positive reference pixels; IoU includes all 77. See `summary.json` and `loro.json` for exact scores and fold thresholds. This is one exploratory run at one fine-tuning learning rate, not evidence of a reliable gain across seeds.

XC validation BCE: 0.185461 versus 0.189267 for the frozen control. The initial head, labels, split and all five epochs' sample-order/CPU-RNG hashes match. Both trained modules update, all other backbone tensors remained bitwise unchanged, and both saved modules match the selected epoch. Total trainable parameters increase from 24,608 to 7,112,480; the inference architecture is unchanged.

Run: `.venv/bin/python scripts/run_last_block_linear_2k.py`

Detached service: `birdsong-last-block-linear2k-20260914.service`.
Progress: `status.json`, `history.json`, and `journalctl --user -u birdsong-last-block-linear2k-20260914.service`.
During evaluation, completed segment files are in `segments/`.
Checkpoints and epoch-boundary resume state are under `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/last_block_linear_2000train_800val_2026-09-14/`.

V100 annotation jobs are untouched. Twins Qwen stays paused; this runner never restarts it.
