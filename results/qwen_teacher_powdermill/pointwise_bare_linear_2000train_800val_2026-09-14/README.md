# Bare linear detector

One `Linear(768,32)` with bias, directly on raw frozen SongMAE-Large output tokens. No detector LayerNorm, hidden projection, positional embeddings, activation, attention or dropout. Trainable parameters: 24,608; the matched LayerNorm+Linear controls have 26,144. The 32 outputs reconstruct each token's 32-mel patch.

The pinned SongMAE revision uses pre-LayerNorm inside each encoder block, then returns the last residual output with no terminal LayerNorm. `SongMAEModel.forward` calls `forward_encoder_inference` with its default `end_of_block` output. The optional normalized `average_top_k` path is not used. Backbone normalization and all backbone weights remain unchanged.

Reuse the completed `pointwise_single_linear_2000train_800val_2026-09-14` controls, exact 2,000-second new-teacher training set and 800-second recording-disjoint XC validation set. Three seeds, five epochs, best validation BCE checkpoint, fixed AdamW learning rate 0.001, weight decay 0.0001, batch 16, hard masks, no target smoothing or TV. Retain exactly the controls' initial linear weights/bias; only remove LayerNorm. Initialization and CPU RNG/sample-order histories are audited. Initial predictions are not expected to match after removing input normalization.

Full Powdermill protocol is unchanged: probability smoothing sigma (2 mel bins, 3 frames), separate single threshold per checkpoint selected on original Recordings 2–4, report Recording 1, continuous-score pixel AP, no morphology or external evaluation. This is a fixed-budget ablation, not evidence that every head is fully converged.

Architecture tag: `pointwise_bare_linear_v1`; use `birdsong_detect_distill.pointwise_bare_linear.load_heads`. Checkpoints contain only `output.weight` and `output.bias`, plus metadata with `layer_norm=False`. `hidden=0` denotes no intermediate projection.

Detached unit: `birdsong-detector-barelinear2k-20260914.service`. Progress: `status.json`, `gpu0.json`, `gpu1.json`, `driver.log`, `logs/`. Final comparison: `summary.json`, including paired bare-minus-LayerNorm differences and three-seed mean/SD. SD is not a dataset confidence interval.

Artifacts live under `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/pointwise_bare_linear_2000train_800val_2026-09-14/`. Existing dependencies, results, figures and annotations are preserved. V100 remains annotating; only Twins GPUs train detectors, and Twins Qwen stays paused afterward. Closing Codex does not stop the service.
