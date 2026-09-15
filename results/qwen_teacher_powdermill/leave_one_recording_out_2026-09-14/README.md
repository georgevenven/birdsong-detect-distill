# Full-Powdermill leave-one-recording-out calibration

All 4 original recordings, 77 five-minute segments, 4,620 five-second windows, 23,100 seconds.
Each segment is scored once, at the threshold selected exclusively from the other original recordings.
Primary aggregation preserves the previous segment-macro definition. Student ± values are sample SD across three training seeds, not confidence intervals across recordings.

| Model / teacher condition | Pixel AP | 2D IoU |
|---|---:|---:|
| Qwen: no reasoning, no axes | 0.242 | 0.076 |
| Qwen: no reasoning, axes | 0.308 | 0.166 |
| Qwen: reasoning, axes | 0.452 | 0.355 |
| Qwen: reasoning + one self-review | 0.457 | 0.361 |
| Qwen: reasoning + two self-reviews | 0.458 | 0.362 |
| SongMAE-Large + bare linear | 0.766 ± 0.004 | 0.512 ± 0.003 |
| SongMAE-Large + transformer | 0.767 ± 0.011 | 0.505 ± 0.009 |

## Per-original-recording scores

Students below are seed 0; JSON and CSV include all seeds.

| Model / condition | Held-out recording | Minutes | Threshold | Pixel AP | 2D IoU |
|---|---|---:|---:|---:|---:|
| Qwen: no reasoning, no axes | Recording_1 | 180 | 0.01 | 0.230 | 0.106 |
| Qwen: no reasoning, no axes | Recording_2 | 70 | 0.01 | 0.307 | 0.033 |
| Qwen: no reasoning, no axes | Recording_3 | 5 | 0.01 | 0.430 | 0.030 |
| Qwen: no reasoning, no axes | Recording_4 | 130 | 0.01 | 0.216 | 0.058 |
| Qwen: no reasoning, axes | Recording_1 | 180 | 0.01 | 0.331 | 0.233 |
| Qwen: no reasoning, axes | Recording_2 | 70 | 0.01 | 0.348 | 0.105 |
| Qwen: no reasoning, axes | Recording_3 | 5 | 0.01 | 0.465 | 0.112 |
| Qwen: no reasoning, axes | Recording_4 | 130 | 0.01 | 0.250 | 0.107 |
| Qwen: reasoning, axes | Recording_1 | 180 | 0.26 | 0.488 | 0.415 |
| Qwen: reasoning, axes | Recording_2 | 70 | 0.26 | 0.463 | 0.294 |
| Qwen: reasoning, axes | Recording_3 | 5 | 0.26 | 0.673 | 0.509 |
| Qwen: reasoning, axes | Recording_4 | 130 | 0.26 | 0.389 | 0.299 |
| Qwen: reasoning + one self-review | Recording_1 | 180 | 0.01 | 0.494 | 0.421 |
| Qwen: reasoning + one self-review | Recording_2 | 70 | 0.01 | 0.467 | 0.304 |
| Qwen: reasoning + one self-review | Recording_3 | 5 | 0.01 | 0.677 | 0.537 |
| Qwen: reasoning + one self-review | Recording_4 | 130 | 0.01 | 0.394 | 0.302 |
| Qwen: reasoning + two self-reviews | Recording_1 | 180 | 0.01 | 0.493 | 0.422 |
| Qwen: reasoning + two self-reviews | Recording_2 | 70 | 0.21 | 0.468 | 0.303 |
| Qwen: reasoning + two self-reviews | Recording_3 | 5 | 0.21 | 0.694 | 0.543 |
| Qwen: reasoning + two self-reviews | Recording_4 | 130 | 0.26 | 0.395 | 0.302 |
| SongMAE-Large + bare linear | Recording_1 | 180 | 0.07 | 0.783 | 0.498 |
| SongMAE-Large + bare linear | Recording_2 | 70 | 0.08 | 0.764 | 0.501 |
| SongMAE-Large + bare linear | Recording_3 | 5 | 0.07 | 0.888 | 0.697 |
| SongMAE-Large + bare linear | Recording_4 | 130 | 0.07 | 0.726 | 0.520 |
| SongMAE-Large + transformer | Recording_1 | 180 | 0.05 | 0.783 | 0.509 |
| SongMAE-Large + transformer | Recording_2 | 70 | 0.06 | 0.748 | 0.479 |
| SongMAE-Large + transformer | Recording_3 | 5 | 0.06 | 0.891 | 0.691 |
| SongMAE-Large + transformer | Recording_4 | 130 | 0.06 | 0.742 | 0.523 |

## Interpretation and limitations

Primary: mean segment AP over 76 reference-positive segments; mean IoU over all 77 segments. Secondary: equal weight to four original-recording summaries. Pooled counts also retained.
Reuse exact continuous-score pixel AP per segment, unchanged by calibration. No pooled pixel AP is inferred from the 101-point threshold counts.
Recording_1 folds reproduce every metric in the existing single-recording reports to numerical precision. Changes in overall AP arise from including Recordings 2–4, not from threshold calibration.
Exploratory development-set reanalysis. Architecture, smoothing and Qwen axes/prompts were already selected using Powdermill; this is not nested model-selection CV or an independent generalization test. Recording_3 contains only five minutes.
Maximum mean segment IoU, lowest threshold breaks exact ties; retain original grids: Qwen 0.01–1.00, students 0.00–1.00, step 0.01.
Fold-specific thresholds are only for this analysis; existing image and external-evaluation thresholds are not changed.
