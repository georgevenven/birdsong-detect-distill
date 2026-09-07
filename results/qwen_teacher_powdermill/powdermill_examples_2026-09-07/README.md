# Powdermill qualitative examples

Open [the contact sheet](00_contact_sheet.png) to compare the four options. Each individual figure is square
(3.5 × 3.5 inches), with large fonts, a 600-dpi PNG (2100 × 2100 pixels), PDF, and editable SVG.
The upper panel shows human bounding boxes; the lower panel shows SongMAE-Large foreground contours.
Both panels display the identical spectrogram, with a shared mel-frequency axis labelled in kHz.

| Option | Description | Original excerpt | PNG | PDF |
| --- | --- | --- | --- | --- |
| A | Sparse vocalizations | Recording_1_Segment_13, 105–110 s | [PNG](A_R1S13_105-110s.png) | [PDF](A_R1S13_105-110s.pdf) |
| B | Repeated song phrase | Recording_1_Segment_08, 80–85 s | [PNG](B_R1S08_80-85s.png) | [PDF](B_R1S08_80-85s.pdf) |
| C | Overlapping frequency bands | Recording_1_Segment_27, 225–230 s | [PNG](C_R1S27_225-230s.png) | [PDF](C_R1S27_225-230s.pdf) |
| D | Dense mixture | Recording_1_Segment_07, 35–40 s | [PNG](D_R1S07_35-40s.png) | [PDF](D_R1S07_35-40s.pdf) |

B is a visually clear repeated-phrase candidate; D shows a denser mixture. C is a harder example with visible errors.
These were selected for varied human-annotation structure and visual variety, not randomly or as
representative estimates of performance. All four candidates are retained. The source species codes, clipped
events, checkpoint/cache hashes, and per-excerpt metrics are recorded in [examples.json](examples.json) and its caches.

The model is the exact self-review 10,000-second checkpoint used in the matched comparisons. The 0.07 probability
threshold comes from Recordings 2–4, not these excerpts. Inference is run on each entire 300-second segment using
the unchanged five-second / 2.5-second-stride pipeline and maximum overlap aggregation, then cropped for display.
There is no Gaussian smoothing, morphology, or small-component removal. Cyan contours outline the raw thresholded
mask, including errors and small detections. Display padding only closes outlines at crop boundaries.
Spectrograms use a fixed −70 to 0 dB range relative to the full segment's peak; the shown frequency range is 20 Hz–16 kHz.
Time on each plot is relative to its excerpt. Only GPU 1 was used, and inference has finished.

Suggested caption (fill in the chosen excerpt):

Qualitative localization on the Powdermill development set. Human time–frequency bounding boxes (top) and
SongMAE-Large predictions (bottom) are shown over the same five-second spectrogram. The frozen-backbone student
was trained on 10,000 seconds of self-reviewed Qwen labels. Cyan contours delineate predictions at the threshold
selected using separate original recordings; no mask post-processing is applied. Frequency is displayed on a mel scale.

Regenerate the figures from cached arrays without using a GPU:

```bash
python scripts/plot_powdermill_examples.py --render-only
```

The initial preparation requires the Powdermill archive, cached backbone, and the original checkpoint. For new
inference, supply new `--out` and `--cache` directories. Re-rendering updates only this new example set.
