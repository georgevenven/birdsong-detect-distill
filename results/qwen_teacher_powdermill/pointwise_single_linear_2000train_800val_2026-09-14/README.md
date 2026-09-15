# LayerNorm plus one linear layer

Head: `LayerNorm(768) -> Linear(768,32)`, applied to each frozen SongMAE-Large token. No hidden projection, detector positions, attention, GELU or dropout. The 32 outputs reconstruct the token's 32-mel patch. Trainable head parameters: 26,144, versus 104,096 for the completed no-position two-linear controls. Backbone parameters and positional encoding remain unchanged.

Three seeds use the exact existing 2,000-second XC training split, 800-second recording-disjoint validation split, hard masks, AdamW 0.001, batch 16 and best validation BCE checkpoint within five epochs. The three existing no-position controls are reused; no new teacher labeling or external-dataset evaluation is performed.

Two affine layers without an intervening nonlinearity can be fused. Each single-layer run starts from the same randomly initialized function as its paired two-layer control: `W = W2 @ W1`, `b = W2 @ b1 + b2`. These are initial random weights, not trained control weights. The fused layer is then optimized directly for five epochs. Initial-function equality is algebraic up to floating-point rounding; initialization hashes and CPU RNG/sample-order histories are audited. This compares parameterization under the same optimizer settings, not an expressivity advantage for the two-layer head.

Powdermill protocol remains fixed: probability Gaussian smoothing sigma (2 mel bins, 3 frames), per-checkpoint threshold maximizing mean IoU on original Recordings 2–4, report original Recording 1, continuous-score pixel AP, no morphology. Three-seed SD is not a dataset confidence interval.

Checkpoint architecture: `pointwise_single_linear_fused_initial_v1`. Use `birdsong_detect_distill.pointwise_single_linear.load_heads`. Legacy `hidden=0` is an explicit sentinel for no hidden projection, not a zero-dimensional layer. Distinct state keys and architecture checks prevent loading it as an old detector.

Detached service: `birdsong-detector-singlelinear2k-20260914.service`. Progress lives in `status.json`, `gpu0.json`, `gpu1.json`, `driver.log` and `logs/`; final comparison and paired differences in `summary.json`.

Artifacts: `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/pointwise_single_linear_2000train_800val_2026-09-14/`. Training checkpoints and initialization audits are retained for recoverable finalization. Existing code dependencies, annotations, checkpoints and figures remain unchanged.

Only Twins GPUs are used. V100 annotations continue uninterrupted, and Twins Qwen stays paused afterward. Closing Codex does not stop the detached service.
