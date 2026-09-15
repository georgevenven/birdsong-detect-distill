# Simplified Qwen prompt: matched 250-window study

| Configuration | Pixel AP | 2D IoU | Pixel precision | Pixel recall | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_plain | 0.2297 | 0.1296 | 0.8260 | 0.0611 | 0.01 |
| direct_axes | 0.3575 | 0.2689 | 0.8218 | 0.1621 | 0.01 |
| reasoning | 0.4900 | 0.4221 | 0.8777 | 0.3587 | 0.01 |
| self_review_1 | 0.4967 | 0.4285 | 0.8810 | 0.3656 | 0.01 |
| self_review_2 | 0.4981 | 0.4307 | 0.8799 | 0.3693 | 0.01 |

Report: 150 windows (750 s) from Recording_1. Calibration: 100 windows (500 s) from Recordings_2–4.
All 77 five-minute segments represented. Same intervals for every condition; no frequency/time smoothing.
Axis selection uses calibration AP only. Reasoning and self-reviews use the selected image format.
These are exploratory teacher results, not student results or full-Powdermill table replacements.
