# Matched Micro/Base/Large figure

Prioritized September 10 at the user's request, gracefully pausing the XC annotation job. New scores are available only once
both per-model `comparison.json` files and the final `summary.json` have been produced.
`birdsong-current-backbones-20260910.service` drains the active XC windows, checks GPU ownership,
frees the idle Qwen server, trains/evaluates Micro and Base, plots the results, then restores the server and resumes XC annotation.
It refuses to interrupt another GPU job. Historical figures and results are preserved.

All three sizes use the frozen self-review labels in
`data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09/manifest.json`:
10,000 s / 2,105 training windows from 249 recordings, plus 800 s / 169 validation windows
from 21 other recordings. These labels do not change as the original XC annotation job expands.

Frozen backbone, 128-dimensional adapter, hard masks, plain BCE, no target smoothing or TV;
AdamW 0.001, weight decay 0.0001, batch 16, dropout 0.1, seed 0, three epochs.
Choose each checkpoint by minimum recording-disjoint XC validation BCE. Same seed/configuration,
but differently sized models need not have identical initial weights or sample permutations.

Use the exact shared Large inference/scoring implementation: 5 s windows, 2.5 s stride,
maximum overlap aggregation, Gaussian probability smoothing (2 mel bins, 3 frames/15 ms),
and one per-model threshold calibrated for mean 2D IoU on the 41 segments of Powdermill Recordings_2–4.
Evaluate all 36 segments / 10,800 s of Recording_1; pixel AP averages its 35 positive segments.
AP uses continuous scores and does not depend on the selected binary threshold.

Reuse the verified current Large result (AP 0.7933325954, IoU 0.5127616533, threshold 0.06)
and Qwen self-review reference (AP 0.4640445065) from the completed stage comparison.
Do not substitute older Micro/Base scores or older 8,235 s reporting coverage.

Outputs on completion: `backbone_pixel_ap.png` (2100 × 2100), PDF, SVG, `backbones.csv`,
`summary.json`, and `caption.txt`; one square 3.5-inch AP-only panel with large fonts.
One seed per size; the plot does not imply statistically significant differences.
