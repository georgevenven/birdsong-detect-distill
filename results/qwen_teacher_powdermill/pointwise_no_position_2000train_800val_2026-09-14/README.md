# Remove detector positional embeddings

Matched ablation of the original linear pointwise head, not the GELU variant. Remove learned frequency/time embeddings only; retain LayerNorm, both linear projections and the frozen SongMAE-Large backbone (including its own positions). Head: `LayerNorm(768) -> Linear(768,128) -> Linear(128,32)`. No GELU, attention or added dropout. Trainable parameters fall from 232,608 to 104,096.

Use exactly the existing new-teacher 2,000-second training split and 800-second recording-disjoint XC validation split. Three seeds, five epochs, best XC validation BCE checkpoint, hard masks, AdamW 0.001, batch 16, no target smoothing or TV. The three completed pointwise controls are reused. Initialize the full control head identically before deleting positional parameters; audit hashes verify shared initialization and all five epochs' sample order/CPU RNG.

Evaluate full Powdermill only: original Recordings 2–4 select each checkpoint's single IoU-maximizing threshold; original Recording 1 supplies report scores. Same Gaussian probability smoothing, continuous-score pixel AP, coverage and no morphology. Do not infer dataset-level statistical significance from three-seed SD.

The isolated `NoPositionHead` has no `time` or `frequency` state keys. Tagged checkpoints require `birdsong_detect_distill.pointwise_no_position.load_heads`; old loaders fail strict loading rather than silently supplying positions. Shared trainer, evaluator, old experiments and running Qwen dependencies remain unchanged.

Detached service: `birdsong-detector-nopos2k-20260914.service`. It waits for GELU to finish, checks the Twins GPUs are free, then trains/evaluates three seeds. V100 remains annotating; Twins Qwen remains paused afterward. Closing Codex does not stop the service.

Progress: `status.json`, `gpu0.json`, `gpu1.json`, `driver.log`, `logs/`. Final comparison and paired differences: `summary.json`.

Checkpoints, initialization audits and sample predictions: `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/pointwise_no_position_2000train_800val_2026-09-14/`. Untagged training checkpoints are retained for recoverable finalization.
