# Label budget and backbone capacity

[PNG for upload](xc_scaling_and_backbones.png) · [PDF](xc_scaling_and_backbones.pdf) · [Editable SVG](xc_scaling_and_backbones.svg)

Square 3.5 × 3.5-inch figure, equal-width panels, shared mask AP axis, and 600-dpi PNG (2100 × 2100 pixels).
Blue identifies the shared SongMAE-Large / 10,000-second configuration. Source scores and hashes are in
[the figure manifest](xc_scaling_and_backbones.json). Existing standalone figures are unchanged.

Suggested caption:

Effect of annotation budget and backbone capacity on Powdermill localization. (a) SongMAE-Large trained on
100, 1,000, and 10,000 seconds of self-reviewed Xeno-Canto labels. (b) SongMAE-Micro, Base, and Large trained
on the same 10,000 seconds of labels. Blue marks the shared Large/10,000-second configuration; dashed lines
show the Qwen self-review teacher. Mask AP is averaged over the same 29 reference-positive segments within
8,235 evaluated seconds of Recording_1. All students use frozen backbones and three training epochs;
points and bars show single runs, not averages over repeated training seeds.

Reproduce without a GPU:

```bash
python scripts/plot_combined_scaling.py
```

The shared checkpoint was evaluated independently for the two studies; its AP differs by less than 0.000001.
Each panel preserves its original score. Full protocols: [label scaling](../self_review_scaling_2026-09-07/README.md)
and [backbone comparison](../backbone_self_review_10000s_2026-09-07/README.md).
