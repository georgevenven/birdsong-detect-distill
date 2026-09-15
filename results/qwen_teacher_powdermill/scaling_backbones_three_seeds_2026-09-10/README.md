# Three-seed two-panel paper figure

Training seeds 0, 1 and 2 for Large at 100 / 1,000 / 10,000 s, and Micro / Base at 10,000 s.
The five verified existing seed-zero runs are reused; ten new runs supply seeds 1 and 2.
Large at 10,000 s uses exactly the same three checkpoints in both panels.

Data membership is fixed: the nested training subsets, 800 s recording-disjoint XC validation,
self-reviewed teacher labels, Powdermill calibration Recordings_2–4, and complete reporting Recording_1.
The seed changes adapter initialization, stochastic training order and dropout, not the data subset.
Same frozen backbone revisions, hard targets/plain BCE, AdamW 0.001, weight decay 0.0001,
batch 16, dropout 0.1 and three epochs. Each run selects its minimum-XC-validation-loss checkpoint,
then calibrates its own IoU threshold on the same 41 Powdermill calibration segments.
Inference remains probability smoothing (2 mel bins, 15 ms) followed by one threshold, with AP
computed from continuous scores on the same 128-mel × 5-ms grid. No morphological cleanup.

Each run's AP averages the 35 positive segments of complete Recording_1 (36 segments, 10,800 s).
The plot shows the arithmetic mean of three run-level AP values with ±1 sample standard deviation
(ddof=1). These are training-seed error bars conditional on fixed data, not recording-sampling
confidence intervals, teacher-annotation uncertainty, or proof of statistical significance.
The teacher baseline remains the same fixed self-reviewed Qwen annotation result.

`birdsong-three-seed-figures-20260910.service` gracefully pauses Qwen, runs one seed per GPU,
creates a separate PNG/PDF/SVG via the existing two-panel plotting code, then resumes Qwen.
Only three dense prediction examples per new run are retained to conserve disk; exact AP,
threshold counts/IoU curves, source provenance and raw probability-array hashes are retained
for every segment. Historical results and figures are not overwritten.

`manifest.json` pins seed-zero reports, checkpoints and label files. On completion, see
`summary.json`, `per_seed.csv`, `caption.txt`, and `xc_scaling_and_backbones.png`.
