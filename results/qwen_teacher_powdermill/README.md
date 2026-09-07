# Partial Qwen evaluation on Powdermill

Results: `partial_1677_windows_2026-09-07.json`. Evaluation ran on CPU using saved annotations.

Coverage is 8,235 seconds (2.2875 hours; 35.65% of Powdermill), from 1,677 completed windows in 30 five-minute segments.
All segments belong to `Recording_1`; only four segments have complete coverage. These are preliminary subset results.
Failed or unfinished windows are excluded for every variant. The JSON preserves the scored intervals, per-segment counts,
pooled metrics, and annotation/evaluator SHA-256 hashes.

## Metrics

Predicted foreground is the union of all `target_vocalization`, `uncertain_vocalization`, and `chorus` boxes.
Human boxes are also unioned on the same 200 Hz × 128 mel-bin grid. Overlapping boxes count once; events are not matched.
IoU measures shared area divided by union area. Area precision measures shared/predicted area; area recall measures shared/reference area.
These three metrics retain every emitted foreground box without confidence filtering. Temporal metrics collapse frequency first.

Mask AP is exact, noninterpolated pixel average precision using the maximum Qwen label confidence at each pixel and zero elsewhere.
It measures confidence ranking across thresholds, not event AP at an IoU cutoff. Qwen's confidence refers to its stated label,
including uncertainty labels; it is not a calibrated per-pixel bird probability.

The table averages each metric over segments. One segment has neither reference nor predicted foreground: its IoU is 1,
its precision/recall are 0 by convention, and its AP is undefined. AP therefore averages 29 segments; other metrics average 30.
The JSON also contains metrics computed from pooled pixel counts.

| Qwen stage | 2D IoU | Area precision | Area recall | Mask AP | Temporal IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 0.3186 | 0.6808 | 0.3335 | 0.3719 | 0.5486 |
| Reasoning | 0.3149 | 0.6973 | 0.3250 | 0.3704 | 0.5387 |
| Self-review | 0.3610 | 0.7769 | 0.3622 | 0.4444 | 0.6264 |
| Shifted review | 0.3576 | 0.7771 | 0.3579 | 0.4418 | 0.6231 |
| Full/adjudication | 0.3570 | 0.7949 | 0.3533 | 0.4449 | 0.6361 |

Self-review improves area overlap and AP. Further review does not improve 2D IoU on this subset, although full adjudication
has the highest temporal IoU. Area recall remains much lower than precision.

The reference regions are rectangles, not traced vocal-energy masks. Powdermill's release README also states that short chips
were excluded and annotated sounds had to be identifiable. Disagreement with those regions is not always an acoustic detection error.
Compare students only on the same scored intervals; existing whole-Powdermill student scores use different coverage.

A completed [matched comparison](matched_self_review_10000s_2026-09-07/README.md) scores the 10,000-second self-review
SongMAE-Large student on exactly these intervals, with its threshold calibrated on the other three original recordings.
The [completed pipeline table](pipeline_table.md) extends this comparison to the initial, shifted-review, and full-pipeline students.
The [self-review scaling experiment](self_review_scaling_2026-09-07/README.md) compares 10, 100, 1,000, and 10,000 seconds of nested training audio on the same intervals.
