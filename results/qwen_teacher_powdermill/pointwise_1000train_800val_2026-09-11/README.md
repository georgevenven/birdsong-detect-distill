# One-layer versus zero-layer predictor at 1,000 seconds

Six fresh runs: frozen SongMAE-Large, predictor width 128, zero/one transformer layers, seeds 0/1/2. This repeats the completed 10k depth comparison at a smaller label budget.

Training uses the existing self-review subset: exactly 1,000 s, 211 windows, 116 recordings, nested unchanged within the original 10k set. The separate XC validation set remains 800 s, 169 windows, 21 disjoint recordings. Dataset membership is fixed across architectures and seeds.

Both heads use hard BCE, no target smoothing or TV, AdamW at 1e-3, weight decay 1e-4, batch 16 and the minimum-validation-BCE checkpoint within five epochs. Existing three-epoch 1k models are not reused as five-epoch controls. Shared weights and CPU RNG initialize identically; paired sample-order/RNG hashes are checked after evaluation.

Powdermill only: full Recording_1 evaluation, threshold selection on Recordings_2–4. Probabilities are Gaussian-smoothed with sigma=(2 mel bins, 3 frames); one threshold maximizes calibration mean IoU. Pixel AP uses continuous smoothed scores before thresholding. No morphology or external-dataset evaluation.

Run configuration and dependencies are frozen in `manifest.json`. The prior training/evaluation code, results and figures are unchanged; the new driver binds separate output paths in its own process. Preflight read every selected real input, checked binary targets, exact duration, label nesting and recording disjointness. No test files were added.

Progress: `status.json`, `gpu0.json`, `gpu1.json` and `logs/`. Final outputs: `pointwise/summary.json`, `comparison.csv` and square `pixel_ap.png/pdf/svg`. AP/IoU summaries report three-seed means and sample SD, not dataset confidence intervals. Only three dense prediction arrays per model are retained; all segment metrics are saved.

Independent service: `birdsong-pointwise-1k-20260911.service`. It survives closing chat/terminal while this computer stays on, but does not automatically restart after failure/reboot. Qwen stays paused after completion. Historical figures are not replaced.
