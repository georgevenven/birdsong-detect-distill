# SongMAE-Large: smoothing ablation

Completed 2026-09-14. Unchanged full-encoder-finetuned Large + linear checkpoint,
3,205 s XC training / 800 s validation, seed 0, selected epoch 3. No retraining.

All Powdermill: 77 five-minute segments, 23,100 s, four original recordings.
Each original recording is evaluated at the threshold maximizing mean segment
IoU on the other three. Raw and smoothed conditions are calibrated separately
on the same 0.00–1.00 grid (step 0.01; lowest threshold breaks ties).

| Inference | Pixel AP | Mean 2D IoU | Pooled precision | Pooled recall |
|---|---:|---:|---:|---:|
| No smoothing + calibrated threshold | 0.756793 | 0.498142 | 0.667925 | 0.656377 |
| Gaussian probability smoothing + calibrated threshold | 0.767669 | 0.520032 | 0.684154 | 0.654389 |

AP is averaged over 76 reference-positive segments; IoU includes all 77.
AP uses continuous scores, before binary thresholding. Neither condition uses
hysteresis, closing, component removal, or hole filling. Gaussian sigma is
2 mel bins and 3 frames (15 ms), applied to the full native probability grid.

| Held-out original recording | Raw threshold | Smoothed threshold |
|---|---:|---:|
| Recording_1 | 0.05 | 0.06 |
| Recording_2 | 0.06 | 0.07 |
| Recording_3 | 0.06 | 0.07 |
| Recording_4 | 0.07 | 0.08 |

Smoothing improves AP by 0.010876 and mean IoU by 0.021890 in this development
comparison. All 77 raw float32 prediction arrays reproduce the original saved
SHA256 hashes bit-for-bit; only smoothing and its separately selected threshold
differ. Full metrics and original Recording_1-only results: `large/comparison.json`.
Per-fold results: `large/loro.json`. Reproduction: `scripts/evaluate_finetuned_unsmoothed.py --sizes large`.

Exploratory single-seed development analysis, not an independent test.
No checkpoint, gallery, or default inference settings changed. Micro stopped at
the user's request after 75 saved segments; those partial results are preserved
but are not reported as complete. Base never started. V100 annotation continued.
