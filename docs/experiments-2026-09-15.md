# Experiment snapshot — 2026-09-15

This is a static checkpoint of completed results, not live job status. Source
reports remain under `results/paper_15000train_2026-09-15` on Twins (a symlink into
`/mnt/birdconv`); large artifacts, runtime JSON, and figures are not pushed to Git.

## Completed training and Powdermill development evaluation

Frozen XC snapshot: 17,875 seconds. Training uses nested budgets up to 15,000 s
(3,000 source recordings; 17,093 teacher boxes). The common validation set has
2,875 s (575 recordings; 3,505 boxes), disjoint by source recording. Newly finished
annotations do not enter these experiments.

Recipe: full SongMAE detection encoder, including CNN and positional embeddings,
plus a bare linear head and sigmoid. Hard-mask BCE, no soft targets or TV. Seed 0,
five epochs, checkpoint selected by minimum XC validation BCE. AdamW encoder LR
1e-5, head LR 1e-3, weight decay 1e-4; effective batch 16 with microbatch 2.
Probability Gaussian smoothing sigma=(2 mel bins, 3 frames), one threshold, no
morphology. Pixel/frame AP uses continuous scores before thresholding.

All 77 Powdermill five-minute segments (23,100 s), with leave-one-original-recording-out
threshold calibration; segment-macro AP/IoU. One seed, no uncertainty estimates.

| Backbone | Training seconds | Pixel AP | 2D IoU |
|---|---:|---:|---:|
| Large | 100 | 0.561 | 0.372 |
| Large | 1,000 | 0.751 | 0.502 |
| Large | 10,000 | 0.758 | 0.508 |
| Large | 15,000 | 0.771 | 0.523 |
| Micro | 15,000 | 0.756 | 0.514 |
| Base | 15,000 | 0.762 | 0.481 |

Source: `runs/{job}/loro.json` and `development.json` in the local report folder.
`scripts/paper_suite_report.py` generates the square two-panel figure. Its existing
PNG/PDF/SVG outputs are local; no figure files or gallery deployment changed here.

## External results completed before pausing

Each model/task gets a separate threshold selected on Powdermill Recordings 2–4,
then held fixed on external datasets. All models retain native inputs and share
the 5-ms scoring grid. SongMAE is the full-encoder Large/15k checkpoint above.

| Dataset | Pixel AP | 2D IoU |
|---|---:|---:|
| WABAD | 0.627 | 0.377 |
| Hawaii | 0.705 | 0.468 |

WABAD: 4,264 retained clips, equal-site macro across 68 sites. Hawaii: 635 clips,
recording macro. Ground truth is the union of reference boxes, not manually
segmented pixels; frequency area is mel-weighted.

| Model | XC-AJ frame AP | Temporal IoU | NIPS4Bplus frame AP | Temporal IoU |
|---|---:|---:|---:|---:|
| SongMAE-Large, 15k | 0.826 | 0.539 | 0.770 | 0.534 |
| YOLO11n released | 0.700 | 0.546 | 0.630 | 0.455 |
| YOLO11l released | 0.676 | 0.556 | 0.590 | 0.464 |

XC-AJ: 288 known-index-disjoint clips (`xcsl` internally). NIPS4Bplus: 674 clips,
with the same ignored annotation intervals; both use recording-macro metrics.
Source: `external/{model}/{dataset}.json` in the local report folder.

Both teacher-fine-tuned YOLO models completed 50 training epochs on the same
15,000/2,875-second split, starting from their released BirdBox checkpoints.
Their external evaluations and BirdCODE's new evaluation have not run yet.
Released YOLO WABAD evaluations paused with 2,866/4,264 nano and 676/4,264 large
clips cached; Hawaii remains pending. Do not describe the full comparison as done.

## Annotation target and safe resume

The user paused evaluation to collect **27,500 total self-reviewed XC seconds**
for a future **25,000 s training + 2,500 s validation** split. That new split is
not created and those models are not trained. Previous validation splits remain
unchanged. Twins and V100 use the same reasoning/axes/one-review annotation
protocol. The viewing machine is not an inference worker.

The detached `birdsong-qwen-budget-27500-20260915.service` counts validated pairs
and gracefully pauses both clients once the target is reached. In-flight calls
may leave a small surplus; the immutable 50k queue is preserved for continuation.

To resume the paper suite later on Twins:

```bash
systemctl --user disable --now birdsong-qwen-budget-27500-20260915.service
systemctl --user enable --now birdsong-paper-15000-20260915.service
```

The suite drains Twins before taking its GPUs and reuses completed training and
per-recording evaluation caches. It does not stop V100. These are machine-local
user services, not portable installation instructions. Do not run GPU experiments
concurrently with Twins annotation.
