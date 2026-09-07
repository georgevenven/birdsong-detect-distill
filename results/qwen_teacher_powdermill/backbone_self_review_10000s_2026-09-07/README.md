# SongMAE backbone comparison: 10,000 s of self-reviewed labels

| Frozen backbone | Backbone parameters | Trainable head parameters | Mask AP ↑ | 2D IoU ↑ | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Micro](micro/comparison.json) | 1,751,873 | 281,888 | 0.649 | 0.421 | 0.03 |
| [Base](base/comparison.json) | 14,889,409 | 315,168 | 0.699 | 0.438 | 0.10 |
| [Large](large/comparison.json) | 98,645,249 | 365,088 | 0.752 | 0.505 | 0.07 |
| Qwen self-review teacher | — | — | 0.444 | 0.361 | All foreground boxes |

[AP figure: PDF](backbone_mask_ap.pdf) · [SVG](backbone_mask_ap.svg) · [PNG](backbone_mask_ap.png) · [Exact scores](summary.json)

At fixed training duration, mask AP increases from Micro to Base to Large. Large gains 0.052 AP over Base
and 0.102 over Micro; all three exceed the teacher. These are single-seed development results, not significance claims.
The figure uses the same square 3.5-inch, large-font Matplotlib layout as the XC scaling figure and shows AP only.

All students use the exact same self-review label file: 10,000 seconds, 2,103 windows, and 266 Xeno-Canto recordings.
The recipe is fixed at three epochs (396 optimizer updates), seed 0, batch size 16, and the same 128-dimensional
adapter design. Input-projection dimensions vary with the backbone. All backbones use 32 × 1 patches and are frozen.
Micro and Base were trained afresh; Large reuses the checkpoint from the existing self-review scaling experiment.
Annotation hashes, training settings, code hashes, model revisions, and parameter counts are in [run.json](run.json).

All three checkpoints were freshly evaluated with identical scoring code and the same completed teacher intervals:
8,235 seconds across 30 segments from Powdermill Recording_1. Each threshold is chosen independently using the same
41 segments from Recordings 2–4. Inference uses five-second windows, 2.5-second stride, maximum overlap aggregation,
and no morphological post-processing. Full details are in the [matched protocol](../matched_self_review_10000s_2026-09-07/README.md).

Mask AP averages the 29 reference-positive segments; IoU averages all 30, with empty-union IoU equal to one.
Large alone predicts no foreground in the all-negative segment. This contributes to its IoU advantage;
the ordering also holds when that segment is excluded. The Large rerun differs from the previous result by less
than 0.000001 in mask AP and 2D IoU, leaving all reported three-decimal values unchanged.
Only physical GPU 1 was used; neither GPU is occupied by this experiment after completion.

Reproduce using cached backbones and the original data paths (override `BIRDCODE_ROOT` if needed):

```bash
bash scripts/run_self_review_backbones.sh
python scripts/plot_self_review_backbones.py results/qwen_teacher_powdermill/backbone_self_review_10000s_2026-09-07
```

The runner skips existing checkpoints and completed comparisons. Generated plots, logs, and checkpoints are
Git-ignored and transferred separately; numeric results and plotting code are versioned.
