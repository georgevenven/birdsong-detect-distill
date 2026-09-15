# Fine-tune transformer blocks, freeze the front end

Train the twelve SongMAE-Large transformer blocks and bare linear detector head. Keep the original pretrained CNN, patch projection, and frequency/time positional embeddings fixed. Unused pretraining decoder and mask tokens also remain frozen. This does not start from the fully fine-tuned checkpoint.

- Same seed 0, 2,000 s XC training / 800 s source-disjoint validation, and original pretrained initialization as the previous ablations.
- Exactly reuse the full-fine-tuning training loop, with a process-local change to the trainable-parameter selector, checkpoint loader and output paths. Existing control scripts/results are untouched.
- Trainable: 85,054,464 transformer parameters + 24,608 head parameters = 85,079,072 total. No inference layers are added.
- AdamW: transformer LR 1e-5, head LR 1e-3, weight decay 1e-4. Logical batch 16, microbatch 2 with eight gradient accumulations, FP32 training. Backbone dropout remains disabled.
- Hard-mask BCE, no target smoothing or TV. Five epochs; choose the best by XC validation BCE, not Powdermill performance.
- Same inference and probability Gaussian smoothing (2 mel bins, 3 frames = 15 ms), followed by one calibrated threshold. No morphology.
- Evaluate all 77 Powdermill segments with leave-one-original-recording-out threshold calibration. Preserve the legacy Recording_1-only report separately. No external benchmark or gallery changes.

Completed successfully. Full-Powdermill leave-one-original-recording-out development scores:

| Seed-0 trainable scope (plus linear head) | Pixel AP | Mean 2D IoU |
|---|---:|---:|
| Backbone fully frozen | 0.761469 | 0.508836 |
| Last transformer block | 0.766917 | 0.513097 |
| Entire detection encoder | 0.757380 | 0.514970 |
| All transformer blocks; front end frozen | 0.764262 | 0.519965 |

Relative to full fine-tuning, freezing the front end increased AP by 0.006882 and mean IoU by 0.004995 in this seed. This variant has the highest mean IoU among these four pilots, while last-block-only has slightly higher AP. It selected epoch 5 by XC validation BCE (0.161103).

AP averages 76 reference-positive segments; mean IoU averages all 77. Pooled precision/recall are 0.692576 / 0.640648 and pooled IoU is 0.498801, which uses different weighting from mean IoU. Exact scores and fold thresholds are in `summary.json` and `loro.json`; `comparison.json` retains the legacy Recording_1 result.

`scope_audit.json` confirms all frozen non-transformer tensors (including CNN, patch projection and positional embeddings) remained bitwise unchanged. All 144 transformer parameter tensors received gradients and updated. The head initialization, sampled window order and CPU RNG match both frozen and full-fine-tuning controls over all five epochs. Saved transformer and head weights both match selected epoch 5 exactly.

This is a single-seed development pilot; visual cleanliness and reference-mask AP are not interchangeable evidence of detection accuracy. No new XC qualitative predictions were generated in this run.

Run: `.venv/bin/python scripts/run_transformer_blocks_linear_2k.py`

Detached service: `birdsong-transformer-blocks-linear2k-20260914.service`.
Progress: `status.json`, `history.json`, `segments/`, and `journalctl --user -u birdsong-transformer-blocks-linear2k-20260914.service`.
Checkpoints and epoch-boundary resume state: `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/transformer_blocks_linear_2000train_800val_2026-09-14/`.
Restore with `transformer_blocks_linear.load_heads`: the checkpoint contains all transformer-block weights and the linear head, with the frozen front end loaded from the pinned original SongMAE revision.

The inherited engine audit field `unused_pretraining_state_sha256` covers **all** non-trainable tensors in this run, including the detection front end; it is not limited to the decoder. `scope_audit.json` records this explicitly.

V100 annotations remain untouched. Twins Qwen stays paused and is not automatically restarted.
