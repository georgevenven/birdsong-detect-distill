# Recording-disjoint teacher–student pipeline ablation

Four cumulative conditions: reasoning-enabled initial prediction, self-review, shifted-context reconciliation,
and conditional adjudication. The unreliable historical direct/no-reasoning control is excluded.

All students use the same new XC split: exactly 10,000 s / 2,105 training windows from 249 recordings,
and 800 s / 169 validation windows from 21 other recordings. The split uses completed matched annotations
from `ablation_2500`; 12.795 s remain unused. No XC-AJ recording IDs occur in either partition.
The old 10k subset is not reused because its leftovers mostly share training recording IDs.
The split manifest records every interval and label-file hash in
`data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09/manifest.json`.

Training: frozen SongMAE-Large at revision `f4eb120734c01ae8bb89f314cf77eefbc2ef8b4e`, 128-dimensional head,
hard foreground masks, plain BCE, no target smoothing or TV penalty. AdamW: learning rate 0.001,
weight decay 0.0001, batch 16, dropout 0.1, seed 0, three epochs. Select the epoch with minimum
validation BCE, weighted by valid pixel count. XC validation selects checkpoints, not detection thresholds.

Student inference: 5 s windows / 2.5 s stride, maximum-probability overlap aggregation, then Gaussian
probability smoothing with sigma 2 mel bins / 3 frames (15 ms), reflect padding, truncate 4.
One calibrated threshold; no hysteresis, closing, component filtering or hole filling.
Teacher pixel scores are maximum foreground-box confidences, with zero outside boxes and no smoothing.

Both teacher and student IoU thresholds are selected independently on the same 41 Powdermill segments
from Recordings_2–4 using a 0–1 grid in steps of 0.01. Maximize mean segment 2D IoU; choose the lowest
threshold on ties. Freeze thresholds before scoring all 36 segments / 10,800 s of Recording_1.
Pixel AP uses continuous scores and averages 35 positive segments; IoU averages all 36, with empty-union IoU 1.
All masks share the 128-bin Slaney-mel (20–16,000 Hz) × 200 Hz time grid. This is a development comparison.

Completed September 9, 2026, 19:49 PDT. All four selected checkpoints are from epoch 3.
See [the updated table](pipeline_table.md), [full results](comparison.json), and [calibration thresholds](calibration.json).
Student thresholds are 0.09 for initial-prediction labels and 0.06 for the three reviewed-label conditions;
all teacher thresholds are 0.01. The self-review student obtains pixel AP 0.793333 and mean 2D IoU 0.512762.
Later stages make small, mixed changes in this single-seed run; these differences do not establish statistical significance.

Run: `bash scripts/run_stage_pipeline.sh`. Historical checkpoints, tables and figures are preserved.
These results should not be mixed with the old subset/final-epoch figure results.
