# SongMAE-Large post-processing, step by step

Five square inspection figures, each with nine panels, saved as 3600 × 3600 PNGs and PDFs.
These are visual diagnostics, not new Powdermill benchmark results or single-column paper layouts.

| Example | Excerpt | PNG | PDF |
|---|---|---|---|
| A: Sparse vocalizations | R1 segment 13, 105–110 s | [PNG](A_R1S13_105-110s_steps.png) | [PDF](A_R1S13_105-110s_steps.pdf) |
| B: Repeated song phrase | R1 segment 08, 80–85 s | [PNG](B_R1S08_80-85s_steps.png) | [PDF](B_R1S08_80-85s_steps.pdf) |
| C: Overlapping frequency bands | R1 segment 27, 225–230 s | [PNG](C_R1S27_225-230s_steps.png) | [PDF](C_R1S27_225-230s_steps.pdf) |
| D: Dense mixture | R1 segment 07, 35–40 s | [PNG](D_R1S07_35-40s_steps.png) | [PDF](D_R1S07_35-40s_steps.pdf) |
| E: Visible hole filling | R1 segment 07, 245–250 s | [PNG](E_R1S07_245-250s_steps.png) | [PDF](E_R1S07_245-250s_steps.pdf) |

## Reading the panels

The cleanup sequence is **b → c → e → f → g → h → i**. Panel d is a separate raw-mask reference,
not an input to the subsequent steps.

- a: Human bounding-box annotations over the spectrogram.
- b: Raw foreground probabilities, on a fixed 0–1 color scale.
- c: Gaussian-smoothed probabilities on the same scale; smoothing is in log-odds space,
  with sigma 2 mel bins and 3 frames (15 ms).
- d: Raw probabilities thresholded at 0.07, the existing development-calibrated operating point.
- e: High-confidence seeds from smoothed probabilities ≥0.45.
- f: Eight-connected propagation of those seeds through smoothed probabilities ≥0.22.
- g: Binary closing with a 3 × 5 mel–time kernel.
- h: Four-connected component filtering: retain at least 32 pixels and a temporal span of at least
  3 frames (15 ms). This uses the full component extent, not the cropped extent.
- i: Fill enclosed background holes, using the existing function's default connectivity.

Cyan outlines show foreground. All spectrogram panels share the same mel frequency axis and
−70 to 0 dB display range, referenced to the full segment's maximum. Added/removed pixel counts
describe the visible crop relative to the preceding binary stage; they are not accuracy scores.

## Provenance and limits

The checkpoint is `artifacts/qwen_ablation_ap_10000/self_review-s10000.pt`, with frozen
`georgeven/songmae-large-32x1` backbone and 10,000 seconds of self-reviewed Qwen labels.
Predictions come from the existing matched-baseline caches: five-second windows, 2.5-second stride,
maximum-probability overlap aggregation. No GPU inference, retraining, or threshold tuning was performed.

Processing was applied to each full 300-second segment before cropping. Source audio, annotation arrays,
checkpoint, cached predictions, and inference-code hashes were checked. The decomposed smoothing and
final masks were bitwise identical to `clean_mask` on all four full segments. `examples.json` records
provenance and changes; `arrays/` preserves the actual displayed intermediate predictions.

A–D retain the earlier qualitative examples. E maximizes the number of pixels changed by closing,
component filtering, and hole filling among other nonoverlapping five-second windows in the same four
segments' matched coverage. It was selected to illustrate morphology, not ground-truth agreement.
None of these examples is a representative performance sample.

The component filter changes no pixels in these five crops. E illustrates hole filling; smoothing and
the stricter hysteresis thresholds visibly remove some weak predictions. A smoother outline does not
by itself establish better detection. Existing AP/IoU tables remain unchanged and use raw probabilities.
The next evaluation should retain the matched reporting intervals and separate calibration recordings;
binary masks alone must not be presented as a confidence-swept mask-AP evaluation.

Methods corrections for this particular checkpoint: its adapter has 365,088 trainable parameters,
and it was trained on all 2,103 supplied windows from 266 recordings for three epochs. It did not use
a 25% training-label validation split or minimum-validation-loss checkpoint selection.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/plot_postprocessing_steps.py \
  --out results/qwen_teacher_powdermill/postprocessing_steps_2026-09-08 --render-only
```

For fresh preparation from the full-segment caches, omit `--render-only` and choose a new output
directory. Existing output directories are protected unless explicit re-rendering is requested.
