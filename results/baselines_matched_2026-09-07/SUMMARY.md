# Corrected matched Powdermill baseline comparison

Report on 8,235 seconds across 30 segments of original Recording_1. Thresholds are selected
on all 41 segments from Recordings 2–4. AP averages 29 positive segments; IoU includes all 30.
Powdermill is development data, including for BirdCODE; these are not unseen-test results.

## Primary: full-band area metrics

| Model | Mask AP | 2D IoU | Temporal occupancy AP | Temporal IoU |
|---|---:|---:|---:|---:|
| [YOLO11n (released BirdBox)](powdermill_final/birdbox.json) | 0.719 | 0.530 | 0.899 | 0.769 |
| [YOLO11n (self-review labels, 10k s)](powdermill_final/qwen_yolo.json) | 0.747 | 0.509 | 0.923 | 0.748 |
| [SongMAE-Large (self-review labels, 10k s)](powdermill_final/songmae.json) | 0.752 | 0.505 | 0.959 | 0.772 |
| [BirdCODE](powdermill_final/birdcode.json) | — | — | 0.926 | 0.759 |

These are box-derived foreground-area metrics, not event AP or bounding-box matching.
SongMAE retains its native full-band input and raw output probabilities; no mask cleanup.
Its full-band AP and IoU reproduce the existing scaling-table result exactly.

## Supplementary: common-band scoring

Only scoring rows intersecting 500–12,000 Hz are selected. Model inputs are unchanged.

| Model | Mask AP | 2D IoU |
|---|---:|---:|
| YOLO11n (released BirdBox) | 0.720 | 0.530 |
| YOLO11n (self-review labels, 10k s) | 0.759 | 0.523 |
| SongMAE-Large (self-review labels, 10k s) | 0.754 | 0.507 |

SongMAE leads full-band and temporal AP; released YOLO leads 2D IoU; teacher-trained YOLO
leads common-band AP. These point estimates do not establish statistical significance or
an architecture-only advantage: pretraining and optimization budgets differ.

## Verification and provenance

- Identical reporting intervals, calibration sources and reference-area counts were verified.
- Both students use all 2,103 windows / 266 recordings / 10,000 s of self-review labels.
- Final YOLO confidence floor: 0.00001; maximum detections: 10,000 (fail if reached).
  [Initial cutoff check](confidence_floor_diagnostic.json) found measurable sensitivity at
  0.0001. The [finer check](confidence_floor_1e-6_diagnostic.json) changed AP by at most
  0.000022 on three preselected calibration segments; this is not full-dataset convergence proof.
- Earlier 0.0001-floor reports remain in `powdermill/`; use `powdermill_final/` for this table.
- [Frozen manifest](manifest.json) and [native-input / split / metric protocol](../../docs/baselines.md).
  BirdCODE/BirdBox native-event paths were checked on diagnostics; full matched event AP and
  external-dataset evaluations have not been run here.
