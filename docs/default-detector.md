# Default detector — selected 2026-09-14

Use fully fine-tuned SongMAE-Large with a bare `Linear(768, 32)` head followed by sigmoid for future detector work unless an experiment explicitly specifies a control.

The user selected this model after inspecting the same held-out XC excerpts with all three models thresholded at 0.09. This is a qualitative preference despite mixed Powdermill benchmark performance, not evidence that it wins on AP. Preserve all existing scores and alternative checkpoints. The XC examples have no human time–frequency reference labels.

## Selected checkpoint and recipe

- Architecture: `songmae_full_encoder_bare_linear_v1`. Fine-tune all 12 encoder blocks, patch projection, both patch convolutions, and native frequency/time positional embeddings. This is **not** the variant that freezes CNN and positional embeddings. Unused pretraining reconstruction components remain frozen.
- Current checkpoint: seed 0, 2,000 s XC training, 800 s source-disjoint XC validation; epoch 5 selected by minimum validation BCE. This is not a 10k/20k model.
- Hard foreground masks, plain BCE, no target softening or TV. AdamW: encoder LR `1e-5`, head LR `1e-3`, weight decay `1e-4`; five epochs, effective batch 16 (microbatch 2 × eight accumulation steps).
- Backbone: `georgeven/songmae-large-32x1`, revision `f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e`.

Checkpoint on Twins:

```text
/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/full_encoder_linear_2000train_800val_2026-09-14/full_encoder_linear_s0.pt
```

SHA-256: `ab8d4be800c68836bfffcd9eda176ad605ec0caaef1e40623c83cb0206f66b84`.

Load the pinned pretrained backbone, then restore **both encoder and head** with [`full_encoder_linear.load_heads`](../src/birdsong_detect_distill/full_encoder_linear.py). The legacy frozen-backbone loader is incompatible. The [completed training run](../results/qwen_teacher_powdermill/full_encoder_linear_2000train_800val_2026-09-14/README.md) records the full protocol and results; [`plot_full_encoder_xc.py`](../scripts/plot_full_encoder_xc.py) demonstrates the correct inference path.

## Inference and reporting

Keep the existing native frontend and normalization, five-second windows with 2.5-second stride, and maximum overlap aggregation. Smooth foreground **probabilities** with Gaussian sigma `(2 mel bins, 3 frames = 15 ms)`, then apply one threshold. No hysteresis or morphological cleanup. Compute pixel AP from continuous smoothed scores before thresholding.

The current checkpoint's reference threshold is **0.09**, calibrated on Powdermill Recordings 2–4, not the XC images. Full-Powdermill leave-one-original-recording-out reporting retains its fold-specific thresholds; future checkpoints require their own development calibration, never test-set tuning.

The common-threshold [XC comparison](../results/xc_three_model_threshold009_2026-09-14/share/README.md) supports the qualitative selection. Historical evaluations and the published gallery are not retroactively changed by this decision.
