# Full SongMAE-Large encoder fine-tuning

Matched single-seed extension of the final-block ablation. Starts from the original pretrained encoder and the same untrained seed-0 bare linear head, not the previously fine-tuned checkpoint.

- Train all 12 encoder blocks, patch projection, both patch convolutions, and native frequency/time positional embeddings: 96,469,248 encoder parameters plus the 24,608-parameter linear head.
- Exclude only unused reconstruction-decoder components and mask tokens. Keep the backbone in evaluation mode to preserve disabled dropout; gradients remain enabled throughout the detection path.
- Same 2,000 s XC training and 800 s source-disjoint validation; hard-mask BCE, no target smoothing or TV.
- AdamW: encoder LR 1e-5, head LR 1e-3, weight decay 1e-4. Five epochs, choose minimum XC validation BCE. No Powdermill checkpoint selection.
- Full-precision training, microbatch 2 with eight gradient accumulations per logical batch of 16; weight microbatch losses by valid time bins. Match the control's initial head and sampled window order. Accumulation can introduce small numerical differences.
- Same Powdermill inference and Gaussian probability smoothing, sigma 2 mel bins / 3 frames (15 ms), followed by one calibrated threshold. Pixel AP uses continuous smoothed probabilities.
- Score all 77 five-minute segments; hold out each original recording in turn and select its threshold on the other three. Retain legacy Recording_1-only scores separately. No external benchmark or gallery changes.

Completed successfully. Full-Powdermill leave-one-original-recording-out results:

| Seed-0 model | Pixel AP | Mean 2D IoU |
|---|---:|---:|
| Fully frozen encoder + linear | 0.761469 | 0.508836 |
| Final block unfrozen + linear | 0.766917 | 0.513097 |
| Full encoder unfrozen + linear | 0.757380 | 0.514970 |

The full-encoder run selected epoch 5, with XC validation BCE 0.160452 (versus 0.185461 for last-block-only and 0.189267 for frozen). Better XC validation BCE did not translate into better Powdermill pixel AP. Compared with last-block-only training, AP decreased by 0.009537 and mean IoU increased by 0.001873. This pilot does not establish a benefit from full fine-tuning.

AP is the mean over 76 reference-positive segments; mean IoU includes all 77. Pooled precision/recall for the full encoder are 0.685420 / 0.638840, and pooled IoU is 0.493999 (different aggregation from mean IoU). Exact fold-specific thresholds and metrics are in `loro.json`; `comparison.json` retains legacy Recording_1 reporting. This is a single-seed development pilot at one learning rate, not a definitive comparison across seeds or tuned learning rates.

Verification: all 152 encoder parameter tensors received gradients and updated; unused pretraining weights remained bitwise unchanged. The initial head, training/validation IDs, target labels and all five sample-order/CPU-RNG hashes match the frozen control. The saved encoder and head both match the selected epoch exactly. Peak allocated GPU memory after the first update was 6.04 GiB. Total trainable parameters: 96,493,856. No inference layers were added.

Run: `.venv/bin/python scripts/run_full_encoder_linear_2k.py`

Detached service: `birdsong-full-encoder-linear2k-20260914.service`.
Progress: `status.json`, `history.json`, `segments/`, and `journalctl --user -u birdsong-full-encoder-linear2k-20260914.service`.
Checkpoints and epoch-boundary resume state: `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/full_encoder_linear_2000train_800val_2026-09-14/`.
Use `full_encoder_linear.load_heads` to restore the entire trained encoder and head for inference; the frozen and final-block loaders are incompatible.

V100 annotations remain untouched. Twins Qwen stays paused; the runner never restarts it.
