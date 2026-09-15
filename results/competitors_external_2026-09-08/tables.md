# External competitor tables

SongMAE external evaluation is intentionally deferred while the detector is developed.

## Time–frequency localization

| Model | WABAD Pixel AP | WABAD 2D IoU | WABAD Pixel precision | WABAD Pixel recall | Hawaii Pixel AP | Hawaii 2D IoU | Hawaii Pixel precision | Hawaii Pixel recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n (released) | 0.552 | 0.347 | 0.567 | 0.539 | 0.541 | 0.369 | 0.498 | 0.609 |
| YOLO11n (our teacher labels) | 0.524 | 0.334 | 0.477 | 0.612 | 0.577 | 0.384 | 0.524 | 0.672 |
| SongMAE-Large (ours) | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## Temporal occupancy

| Model | XC-AJ Frame AP | XC-AJ Temporal IoU | XC-AJ Frame precision | XC-AJ Frame recall | NIPS4Bplus Frame AP | NIPS4Bplus Temporal IoU | NIPS4Bplus Frame precision | NIPS4Bplus Frame recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BirdCODE | 0.836 | 0.526 | 0.503 | 0.947 | 0.657 | 0.545 | 0.574 | 0.883 |
| YOLO11n (released) | 0.700 | 0.546 | 0.563 | 0.836 | 0.630 | 0.455 | 0.629 | 0.783 |
| YOLO11n (our teacher labels) | 0.735 | 0.466 | 0.445 | 0.957 | 0.704 | 0.441 | 0.546 | 0.852 |
| SongMAE-Large (ours) | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## Protocol notes

WABAD: 4,264 primary recordings, 68 sites. Three of the 4,267 candidates contain inverted annotation bounds and are excluded uniformly; zero-extent references contribute zero area. AP/IoU use within-site recording means followed by equal site weighting; precision/recall pool pixels within each site, then average sites equally. Hawaii: 635 original FLAC recordings; AP/IoU are recording means and precision/recall pool pixels. Full 20–16,000 Hz, 128-bin mel area is primary; common-band and per-site alternatives remain in the complete reports.

XC-AJ: the frozen 288-recording known-index-disjoint subset, not the complete 967-recording dataset. NIPS4Bplus: 674 annotated clips, non-birds negative, Unknown intervals ignored. Frame AP/IoU are recording means; frame precision/recall pool counts. All models use the same 200-Hz scoring grid, without changing their native input or output resolution.

All operating thresholds are fixed from Powdermill Recordings 2–4. Released YOLO: pixel 0.05, frame 0.04; teacher-trained YOLO: 0.01 for both; BirdCODE: frame 0.20. AP uses continuous foreground scores. Empty-union IoU is one; AP excludes recordings without positive reference area. These are box-derived area/occupancy metrics, not box AP@0.5 or event F1.

The Hawaii CSV contains 196 zero-duration boxes; they remain in the source but contribute zero reference area. Faint/unidentified calls may be unannotated. Broader pretraining exposure is not fully certified; see [Hawaii's original release](https://zenodo.org/records/7078499) and the [BirdBox paper](https://arxiv.org/html/2606.10407v1).

CSV/TSV files retain full precision. `table_sources.json` records the report checksums. Historical results and the manuscript's older teacher-stage student values are not overwritten.
