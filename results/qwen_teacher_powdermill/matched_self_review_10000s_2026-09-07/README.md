# Matched Qwen–SongMAE comparison

Compare Qwen self-review annotations with `artifacts/qwen_ablation_ap_10000/self_review-s10000.pt`:
SongMAE-Large trained for three epochs on 10,000 seconds of self-reviewed XCL labels (2,103 windows, 266 recordings).

| Model | 2D IoU | Area precision | Area recall | Mask AP | Temporal IoU | Temporal AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen self-review | 0.3610 | 0.7769 | 0.3622 | 0.4444 | 0.6264 | 0.8593 |
| SongMAE-Large | 0.5050 | 0.5699 | 0.7391 | 0.7515 | 0.7731 | 0.9585 |

SongMAE gains 0.1440 mean 2D IoU and 0.3071 mask AP. It recovers much more reference area, at lower area precision.
Its IoU improves on 27 segments, ties on one empty segment, and declines on two. Its mask AP improves on all 29
segments with reference positives. These segment outcomes are correlated and are not a statistical significance claim.

Both models are scored on exactly the completed intervals saved in `../partial_1677_windows_2026-09-07.json`:
8,235 seconds across 30 segments from `Recording_1`. Failed and unfinished Qwen windows are excluded for both models.
The scorer checks interval durations and reference-positive counts against the saved teacher evaluation.

Both use the same 200 Hz × 128 mel-bin reference masks, exact noninterpolated pixel AP, and segment-mean aggregation.
Foreground includes target, uncertain, and chorus labels. There is no instance matching or morphological post-processing.
Qwen retains all emitted foreground boxes; their maximum label confidence supplies its AP scores.
SongMAE retains its existing five-second windows, 2.5-second stride, and maximum-probability overlap aggregation.
Thus evaluation coverage and scoring are matched; each model retains its native inference procedure.

The student probability threshold, 0.07, is selected by maximum mean 2D IoU over 101 thresholds on 41 segments from `Recording_2`,
`Recording_3`, and `Recording_4`. It is saved in `calibration.json` before scoring `Recording_1`.
AP does not depend on this operating threshold. Powdermill previously informed development, so this is a development
comparison, not a new held-out generalization result.

One comparison segment has no reference positives. AP omits it and averages 29 segments; the other metrics average 30.
An empty union has IoU 1; undefined precision/recall are 0. `comparison.json` also preserves pooled metrics,
per-segment counts, coverage intervals, model/code hashes, and the cached backbone revision.

The comparison uses physical GPU 1 exclusively. Reproduce with `scripts/compare_qwen_songmae.py`, passing the teacher
results, checkpoint, BirdCODE data root, and a new output directory.
