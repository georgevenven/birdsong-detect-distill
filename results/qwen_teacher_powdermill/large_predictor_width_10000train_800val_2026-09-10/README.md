# Predictor width with a fixed Large backbone

Compare predictor dimensions 128, 384, and 768 on the same frozen SongMAE-Large backbone.
Train only the missing d384 condition; reuse the matching d128 and d768 checkpoints.
One seed (0), five-epoch budget, minimum XC validation BCE for checkpoint selection.
The reused d128 checkpoint was saved during a 15-epoch run but is also the minimum
within its first five epochs (epoch 3); its learning-rate schedule was constant.

Exactly 10,000 s of existing self-reviewed XC training labels and a recording-disjoint
800 s XC validation set. Hard-mask BCE, no target softening or TV penalty; AdamW lr
0.001, weight decay 0.0001, batch 16, dropout 0.1. Preserve one four-head transformer
layer and feedforward width 2*d. Larger d changes total predictor capacity, not only
the input projection. The new d384 predictor has 1,878,560 trainable parameters.
Seed and dataset membership are fixed; head shapes change RNG consumption and hence
minibatch permutations. This is a single-seed exploratory comparison.

Powdermill only: Recordings_2–4 calibrate each model's single threshold for mean 2D IoU;
complete Recording_1 supplies reporting scores. Gaussian probability smoothing remains
2 mel bins / 15 ms, without morphology. Pixel AP uses continuous smoothed probabilities
and averages the 35 positive segments; IoU averages all 36 reporting segments.
No external datasets, additional Qwen calls, or backbone finetuning.

`manifest.json` freezes data, source reports/checkpoints, previous figures, and shared
implementation hashes. The runner trains, evaluates all 77 calibration/reporting segments,
then writes `comparison.csv`, `summary.json`, and `large_predictor_width_ap.{png,pdf,svg}`.
Only three dense prediction examples are retained; every segment keeps exact scores and
probability-array hashes. Historical results are not overwritten. Qwen remains paused.

Runner: `scripts/run_large_predictor_width.sh`.
Service: `birdsong-large-predictor-width-20260910.service`.
