# Current two-panel paper figure

Reuses `scripts/plot_combined_scaling.py`, with (a) the 100 / 1,000 / 10,000 s label-budget curve,
(b) Micro / Base / Large at 10,000 s, and one teacher legend above both panels.
The square 3.5-inch figure is exported as 600-dpi PNG, PDF, and SVG, without the manuscript's surrounding table or caption.

The old small-budget points are not compatible with the current reporting coverage and checkpoint-selection protocol.
Two new Large adapters therefore use nested prefixes of the frozen seed-17 ordered 10,000 s self-review training set.
The last window is clipped to obtain exactly 100 or 1,000 s. All budgets reuse the same separate 800 s XC validation set;
validation recordings never enter training. Selection does not inspect Powdermill scores or annotation quality.

Same simplified training/inference as the current backbone and stage runs: frozen backbone, hard targets, plain BCE,
AdamW 0.001, weight decay 0.0001, batch 16, seed 0, three epochs with minimum XC validation loss selecting the checkpoint.
Gaussian probability smoothing (2 mel bins, 15 ms), no other cleanup; per-model IoU calibration on Powdermill Recordings_2–4.
Report full Recording_1 (36 segments / 10,800 s), with exact continuous mask AP averaged over its 35 positive segments.
The blue 10,000 s point and Large bar reuse the same verified checkpoint/AP; teacher AP is identical in both panels.

`birdsong-current-scaling-20260910.service` gracefully pauses Qwen, runs the two small-budget experiments,
builds the figure, then restores Qwen. Existing tables, checkpoints, and figures remain untouched.
Outputs are valid once both `s100/comparison.json` and `s1000/comparison.json` and the final plot exist.
See `caption.txt` and `figure_values.csv` on completion. The caption should not claim monotonically better AP with backbone
size: in the current single-seed comparison Base slightly exceeds Large in AP, while Large has higher calibrated IoU.
