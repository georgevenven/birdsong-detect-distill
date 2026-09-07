# Matched annotation-stage comparison

| Qwen annotation pipeline | Teacher mask AP ↑ | Teacher 2D IoU ↑ | Student mask AP ↑ | Student 2D IoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| [Initial prediction](matched_reasoning_10000s_2026-09-07/comparison.json) | 0.370 | 0.315 | 0.730 | 0.474 |
| [With self-review](matched_self_review_10000s_2026-09-07/comparison.json) | 0.444 | 0.361 | 0.752 | 0.505 |
| [With self-review and independent shifted-context review](matched_shifted_review_10000s_2026-09-07/comparison.json) | 0.442 | 0.358 | 0.750 | 0.505 |
| [Full pipeline including conditional adjudication](matched_full_10000s_2026-09-07/comparison.json) | 0.445 | 0.357 | 0.748 | 0.498 |

Table X. Effect of teacher annotation stages on teacher and downstream student localization on the Powdermill development subset.
Each student uses a frozen SongMAE-Large backbone and the same 10,000 s of training audio annotated with the corresponding pipeline.
All configurations are evaluated on identical completed intervals totaling 8,235 s across 30 segments from Recording_1.
Student operating thresholds are selected using the other three original recordings; teacher IoU retains all emitted foreground boxes.
Mask AP summarizes pixel-level precision–recall performance across confidence thresholds.

Initial prediction is the reasoning-enabled first pass, not the separate no-reasoning baseline.
The students share all 2,103 training windows from 266 XCL recordings and the same three-epoch training procedure.
Thresholds are 0.07 for initial prediction, self-review, and shifted review, and 0.06 for the full pipeline.
Each threshold maximizes mean 2D IoU on the same 41 calibration segments, using a 101-threshold grid, before comparison scoring.

Scores are segment means on the same 200 Hz × 128 mel-bin grid. AP averages 29 segments with reference positives;
IoU averages all 30, assigning 1 to an empty prediction/reference union. AP is exact and noninterpolated for both models.
SongMAE retains five-second windows with 2.5-second stride, maximum-probability overlap aggregation, and no morphological cleanup.
Each linked result records coverage, per-segment counts, checkpoint/code hashes, and its calibration procedure.

Self-review provides the largest gains in this run. Additional review gives no consistent improvement over self-review alone.
These are development results from one original recording; small differences between stages are descriptive, not significance claims.
Only physical GPU 1 was used for student evaluation; no models were retrained.
