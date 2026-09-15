# Matched external evaluation: current students and released baselines

## Table 2

| Model | WABAD Pixel AP | WABAD 2D IoU | Hawaii Pixel AP | Hawaii 2D IoU |
|---|---:|---:|---:|---:|
| YOLO11n (released) | 0.552 | 0.347 | 0.541 | 0.369 |
| YOLO11l (released) | 0.549 | 0.354 | 0.546 | 0.381 |
| YOLO11n (our teacher labels) | 0.528 | 0.333 | 0.576 | 0.368 |
| SongMAE-Large (ours) | 0.604 | 0.363 | 0.669 | 0.435 |

## Table 3

| Model | XC-AJ Frame AP | XC-AJ Temporal IoU | NIPS4Bplus Frame AP | NIPS4Bplus Temporal IoU |
|---|---:|---:|---:|---:|
| BirdCODE | 0.836 | 0.526 | 0.657 | 0.545 |
| YOLO11n (released) | 0.700 | 0.546 | 0.630 | 0.455 |
| YOLO11l (released) | 0.676 | 0.556 | 0.590 | 0.464 |
| YOLO11n (our teacher labels) | 0.754 | 0.472 | 0.702 | 0.478 |
| SongMAE-Large (ours) | 0.897 | 0.468 | 0.823 | 0.521 |

Both current students use identical self-reviewed Qwen labels: 10,000 s of XC training audio and 800 s of recording-disjoint XC validation audio.
SongMAE uses hard-mask BCE, Gaussian probability smoothing and a single threshold. YOLO retains its native box objective, spectrogram input, initialization and validation mAP checkpoint selection.
Each model’s pixel and frame thresholds are independently calibrated on the same 41 Powdermill segments from Recordings 2–4, then frozen before external evaluation.
WABAD: 4,264 recordings across 68 sites, site-macro scores. Hawaii: 635 recordings, recording-macro scores. Pixel masks use 128 mel bins over 20–16,000 Hz.
XC-AJ: the frozen 288-recording known-index-disjoint subset. NIPS4Bplus: 674 annotated clips, non-birds negative, Unknown intervals ignored. Temporal scores use the common 5-ms grid.
AP uses continuous scores; IoU uses fixed calibrated thresholds. These are area/occupancy metrics, not event-matching AP. CSV/TSV files include full-precision precision and recall.
Released YOLO and BirdCODE results are reused unchanged after coverage, source-hash, reference-mask and calibration checks. Broader pretraining exposure is not fully certified.

YOLO11l is the released BirdBox large checkpoint, without teacher-label retraining. It uses native BirdBox preprocessing, NMS and the same Powdermill calibration recordings; all earlier model results remain unchanged.
