# Post-processed Powdermill examples

Open [the contact sheet](00_contact_sheet.png). These are the same four excerpts, checkpoint, spectrograms,
and human boxes as the [raw versions](../powdermill_examples_2026-09-07/README.md), which remain unchanged.
Each individual figure is 3.5 × 3.5 inches with a 600-dpi PNG, PDF, and editable SVG.

| Option | Content | PNG | PDF |
| --- | --- | --- | --- |
| A | Sparse vocalizations | [PNG](A_R1S13_105-110s.png) | [PDF](A_R1S13_105-110s.pdf) |
| B | Repeated song phrase | [PNG](B_R1S08_80-85s.png) | [PDF](B_R1S08_80-85s.pdf) |
| C | Overlapping frequency bands | [PNG](C_R1S27_225-230s.png) | [PDF](C_R1S27_225-230s.pdf) |
| D | Dense mixture | [PNG](D_R1S07_35-40s.png) | [PDF](D_R1S07_35-40s.pdf) |

The bottom panel now uses the existing `clean_mask` defaults, applied to each full 300-second segment before cropping:

- Gaussian smoothing in log-odds space: 2 mel bins × 3 frames (15 ms).
- Eight-connected hysteresis: high threshold 0.45, low threshold 0.22.
- A 3 × 5 mel-time closing operation, followed by four-connected component filtering.
- Remove components below 32 pixels or three frames in width, then fill holes.

These are repository defaults, not thresholds selected to improve these examples. They replace the raw figure's
single 0.07 threshold. A and B provide clear qualitative examples; C and D show the recall cost of the stricter
thresholds, with weaker vocalizations no longer detected. All four candidates are retained without reselection.

The existing AP/IoU benchmark tables are unchanged and still describe unprocessed predictions. These figures
must be captioned as post-processed. In [examples.json](examples.json), `raw_threshold` preserves the comparison's
original threshold, while `postprocessing` records the settings actually displayed. Per-excerpt mask AP uses
Gaussian-smoothed probabilities; the other per-excerpt metrics use the final cleaned masks. None are aggregate results.

Suggested caption:

Qualitative localization on the Powdermill development set. Human bounding-box annotations (top) and
post-processed SongMAE-Large predictions (bottom) are shown over the same five-second spectrogram. The student
was trained on 10,000 seconds of self-reviewed Qwen labels with a frozen backbone. Cyan contours delineate
foreground regions after log-odds smoothing, hysteresis thresholding, morphological closing, small-component
removal, and hole filling. Frequency is displayed on a mel scale.

Re-render from the cached full-context predictions and cleaned masks without a GPU:

```bash
python scripts/plot_powdermill_examples.py --postprocess --render-only
```

Initial preparation uses `--postprocess` without `--render-only`; completed outputs are protected from replacement.
Only GPU 1 was used for preparation and is now free. The user's GPU 0 task was left alone.
