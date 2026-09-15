# Pointwise GELU ablation

Only change: insert exact GELU after the projected features plus learned frequency/time embeddings, before the output linear layer. No extra parameters or dropout. Frozen SongMAE-Large, width 128, zero attention layers.

Reuse the exact completed `new_teacher_2000train_800val_2026-09-14` split and three linear controls. Train GELU seeds 0, 1, 2 for five epochs; choose minimum validation BCE on the same separate 800 seconds of XC. Training uses all 2,000 seconds, hard masks, AdamW 0.001, batch 16, no soft targets or TV. Initial parameter values, sample order, CPU RNG, parameter count, checkpoint settings and evaluation coverage are checked against paired controls.

Full Powdermill only: calibrate a separate single threshold for each checkpoint on original Recordings 2–4; report Recording 1. Identical probability smoothing, sigma (2 mel bins, 3 frames), continuous-score pixel AP, no morphology. No external-dataset evaluation.

The isolated `GeluPointwiseHead` wraps the existing output linear layer with GELU. Its distinct `output.1.*` state keys prevent accidental loading as an ordinary linear head. Checkpoints are explicitly tagged `pointwise_gelu_after_position_v1`; use `birdsong_detect_distill.pointwise_gelu.load_heads` to load them. The driver selects this factory within its training/evaluation subprocesses; all existing shared source files remain unchanged.

Detached unit: `birdsong-detector-gelu2k-20260914.service`. Inspect `driver.log`, `logs/`, `status.json`, and `gpu0.json`/`gpu1.json`. Final results: `summary.json`, including three-seed mean/SD and paired GELU-minus-linear differences. SD describes seed variability, not uncertainty from dataset sampling.

Checkpoints and sample predictions are under `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/pointwise_gelu_2000train_800val_2026-09-14/`. Untagged `.training.pt` files are retained for recoverable finalization. Existing experiments and figures are not overwritten.

Use only the two Twins GPUs. V100 annotations continue uninterrupted; Twins Qwen stays paused after this experiment. Closing Codex or the terminal does not stop the detached service.
