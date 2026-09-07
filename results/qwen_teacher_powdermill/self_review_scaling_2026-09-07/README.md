# Self-review label scaling

Frozen SongMAE-Large students trained on nested subsets of self-reviewed Xeno-Canto annotations:

| Training audio (s) | Windows | Recordings | Mask AP ↑ | 2D IoU ↑ | Threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| [10](../matched_self_review_10s_2026-09-07/comparison.json) | 2 | 2 | 0.219 | 0.183 | 0.15 |
| [100](../matched_self_review_100s_2026-09-07/comparison.json) | 21 | 18 | 0.303 | 0.207 | 0.09 |
| [1,000](../matched_self_review_1000s_2026-09-07/comparison.json) | 209 | 112 | 0.650 | 0.426 | 0.07 |
| [10,000](../matched_self_review_10000s_2026-09-07/comparison.json) | 2,103 | 266 | 0.752 | 0.505 | 0.07 |

[Figure: PNG](scaling.png) · [PDF](scaling.pdf) · [SVG](scaling.svg) · [Exact scores](scaling.json)

Single-column square figure: [PDF](scaling_single_column.pdf) · [SVG](scaling_single_column.svg) · [PNG](scaling_single_column.png).
This is a 3.5 × 3.5-inch Matplotlib figure with 10–11 pt tick/axis fonts, embedded PDF fonts, editable SVG text,
and a 600-dpi PNG. It shows only mask AP at 100, 1,000, and 10,000 training seconds.
Blue circles show SongMAE-Large; the dashed line shows Qwen self-review mask AP.
Generated figures and machine-local `song-detect-paper/` work are ignored by Git and transferred separately.

Reproduce from the three displayed matched comparisons (no GPU needed):

```bash
python scripts/plot_self_review_scaling.py \
  --result 100 results/qwen_teacher_powdermill/matched_self_review_100s_2026-09-07/comparison.json \
  --result 1000 results/qwen_teacher_powdermill/matched_self_review_1000s_2026-09-07/comparison.json \
  --result 10000 results/qwen_teacher_powdermill/matched_self_review_10000s_2026-09-07/comparison.json \
  --out results/qwen_teacher_powdermill/self_review_scaling_2026-09-07/scaling_single_column
```

Performance improves most between 100 and 1,000 seconds and continues to improve at 10,000 seconds.
The 1,000-second student already exceeds the Qwen self-review teacher (mask AP 0.444, 2D IoU 0.361) on this subset.
The dashed figure lines show those teacher scores.

All models use the same 8,235 evaluated seconds across 30 segments of Powdermill Recording_1.
Each operating threshold is selected independently using the same 41 segments from Recordings 2–4 and the same 101-threshold grid.
AP is exact and noninterpolated; it averages the 29 segments with reference positives. IoU averages all 30 segments.
Other details follow the [matched comparison protocol](../matched_self_review_10000s_2026-09-07/README.md).

The 100-, 1,000-, and 10,000-second checkpoints and labels come from `qwen_ablation_ap_10000`.
The new 10-second model uses the first two complete windows from that 100-second subset: XC583859 and XC535066.
Its annotations are `data/annotations/xcl/self_review_scaling_2026-09-07/10s.jsonl`, and its checkpoint is
`artifacts/self_review_scaling_2026-09-07/self_review-s10.pt`. Nesting and event-label equality were verified across all budgets.
Ten thousand seconds (2.78 hours) is the largest trained subset here; the underlying annotation snapshot contains 10,812.795 seconds.

Every model uses the same three-epoch training recipe. Consequently, optimization effort also grows with the dataset:
the 10-second model receives only three optimizer updates. This curve does not isolate label quantity from training-step count.
At the selected thresholds, both the 10- and 100-second models predict bird presence throughout the scored time;
their temporal IoU of 0.624 should not be interpreted as useful temporal localization.
These are single-run development results from one original Powdermill recording. Only GPU 1 was used.
