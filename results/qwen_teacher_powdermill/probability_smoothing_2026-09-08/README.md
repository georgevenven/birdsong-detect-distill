# Raw versus lightly smoothed SongMAE probabilities

Gaussian smoothing followed by one calibrated threshold improves mask AP and mean 2D IoU in this matched
Powdermill development comparison. Precision increases slightly and recall decreases slightly.

| Prediction | Threshold | Mask AP | Mean 2D IoU | Pooled area precision | Pooled area recall |
|---|---:|---:|---:|---:|---:|
| Raw probabilities | 0.07 | 0.751512 | 0.505036 | 0.599650 | 0.724334 |
| Gaussian-smoothed probabilities | 0.08 | 0.768077 | 0.516322 | 0.619312 | 0.719671 |

Absolute changes: AP +0.016565, IoU +0.011286, precision +0.019662, recall −0.004663.
These descriptive development results do not establish statistical significance or unseen-dataset generalization.
The existing production cleanup function, benchmark tables, and earlier figures were not changed.

## Five visual comparisons

Each square figure shows human boxes (top), raw predictions at 0.07 (middle), and probability-smoothed
predictions at 0.08 (bottom). Thin cyan contours come from the actual evaluated binary masks; there is
no contour-only smoothing or additional morphological cleanup. PNGs are 2560 × 2560; PDFs are also supplied.

| Example | PNG | PDF |
|---|---|---|
| A: Sparse vocalizations | [PNG](figures/A_R1S13_105-110s_comparison.png) | [PDF](figures/A_R1S13_105-110s_comparison.pdf) |
| B: Repeated song phrase | [PNG](figures/B_R1S08_80-85s_comparison.png) | [PDF](figures/B_R1S08_80-85s_comparison.pdf) |
| C: Overlapping frequency bands | [PNG](figures/C_R1S27_225-230s_comparison.png) | [PDF](figures/C_R1S27_225-230s_comparison.pdf) |
| D: Dense mixture | [PNG](figures/D_R1S07_35-40s_comparison.png) | [PDF](figures/D_R1S07_35-40s_comparison.pdf) |
| E: Earlier morphology-change example | [PNG](figures/E_R1S07_245-250s_comparison.png) | [PDF](figures/E_R1S07_245-250s_comparison.pdf) |

All five excerpts are unchanged from the previous stage-by-stage visualization; no examples were reselected.
They are illustrative, not a representative sample. The aggregate table above scores the complete matched subset.

## Protocol

- Same frozen SongMAE-Large backbone and 10,000-second self-review-label checkpoint as the earlier comparison.
- Smoothing operates directly on float32 probabilities, not log odds: sigma 2 mel bins × 3 frames (15 ms),
  reflective boundaries, kernel truncation at 4 sigma. See [SciPy's Gaussian filter documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).
- Filtering uses each full cached spectrogram, including its STFT endpoint frame, before cropping to the
  actual audio duration or scoring intervals. No smoothing across disconnected excerpts or recording boundaries.
- No hysteresis, closing, component removal, or hole filling in either condition. No smoothing-strength search.
- Each condition independently selects the threshold maximizing mean segment 2D IoU over 41 segments
  (12,300 seconds) from original Recordings 2–4, using the same 101-point 0–1 grid. Thresholds are saved
  before reporting-set scoring. There is no per-excerpt or reporting-set threshold tuning.
- Reporting uses the frozen 8,235 seconds of teacher-covered intervals in 30 segments from Recording_1.
  All scoring uses the full 20–16,000 Hz band on a 128-mel × 200-Hz grid.
- Mask AP is exact, noninterpolated AP of continuous scores, before thresholding; it averages the 29 segments
  with reference positives. IoU averages all 30 segments, with empty-union IoU = 1. Precision and recall
  pool pixel counts across those same intervals. These are area metrics, not event AP or box matching.
- Raw per-segment AP, TP/FP/FN curves, IoU curves, calibrated threshold, and aggregate scores exactly reproduce
  the existing matched baseline. Full-segment masks used for the figures independently reproduce the
  evaluated counts at the frozen thresholds. Existing audio, label, code, and prediction hashes were checked.

Calibration IoU also increased, from 0.515272 to 0.529450. Results use one model checkpoint and one original
reporting recording; the next step is to validate a frozen choice on separate datasets. Other checkpoints
in a scaling/backbone comparison should use the same declared post-processing and calibration procedure.

Full precision, provenance, coverage, per-segment curves and counts: [comparison.json](comparison.json).
Threshold selection: [calibration.json](calibration.json). Figure arrays and checksums: [figures/examples.json](figures/examples.json).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/evaluate_songmae_smoothing.py --out NEW_DIRECTORY --workers 4
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/plot_smoothing_comparison.py --report NEW_DIRECTORY/comparison.json
```

Existing outputs are protected. Add `--render-only` to the plotting command to redraw this experiment
from its saved figure arrays. Both stages run on CPU; no GPU inference or retraining is needed.
