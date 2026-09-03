# Exploratory two-dimensional localization

YOLO11n and SongMAE Large 32×1 were compared directly against human time-frequency boxes. Reference Hz bounds were projected onto the 128-bin mel axis used by both models. The pilot uses 16 recordings from each dataset; WABAD is spread across 16 sites. These are directional estimates, not full-dataset results.

| Dataset | Model | 2D AP @ .2 | 2D AP @ .5 | Best F1 @ .2 | Best F1 @ .5 |
|---|---|---:|---:|---:|---:|
| Powdermill | YOLO11n + Qwen | 0.143 | 0.016 | 0.261 | 0.060 |
| Powdermill | SongMAE Large 32×1 | **0.273** | **0.091** | **0.435** | **0.241** |
| XCSL | YOLO11n + Qwen | 0.260 | 0.155 | 0.339 | 0.249 |
| XCSL | SongMAE Large 32×1 | **0.852** | **0.625** | **0.833** | **0.711** |
| WABAD | YOLO11n + Qwen | 0.126 | **0.052** | 0.218 | 0.144 |
| WABAD | SongMAE Large 32×1 | **0.195** | 0.049 | **0.393** | **0.202** |

SongMAE wins all six best-F1 comparisons and five of six AP comparisons. Its strict WABAD AP is effectively tied with YOLO on this small sample, while its best strict-IoU F1 remains substantially higher. The especially large XCSL gap is consistent with SongMAE's stronger temporal transfer, but its magnitude should be confirmed on more recordings.

SongMAE boxes are connected components from the deployed hysteresis, smoothing, closing, hole filling, and small-component removal pipeline; their confidence is the peak smoothed probability. YOLO uses its native boxes and confidence. Both use 51 confidence thresholds and one-to-one Hungarian matching at 2D IoU 0.2 and 0.5.
