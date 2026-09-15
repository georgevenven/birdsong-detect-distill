# Full Powdermill: simplified Qwen prompt

| Configuration | Pixel AP | 2D IoU | Pixel precision | Pixel recall | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_plain | 0.2298 | 0.1062 | 0.8209 | 0.0628 | 0.01 |
| direct_axes | 0.3308 | 0.2332 | 0.8287 | 0.1658 | 0.01 |
| reasoning | 0.4882 | 0.4145 | 0.8502 | 0.3735 | 0.26 |
| self_review_1 | 0.4942 | 0.4210 | 0.8522 | 0.3834 | 0.01 |
| self_review_2 | 0.4930 | 0.4222 | 0.8532 | 0.3864 | 0.01 |

All 4,620 windows / 23,100 seconds are annotated for each completed condition.
Report: full Recording_1 (2,160 windows, 10,800 s). Calibrate: full Recordings_2–4 (2,460 windows, 12,300 s).
Axes fixed from the pilot. Same prompts, numbered reviews, seeds for reused windows, and scoring method.
250 pilot outputs reused per condition with original provenance. No historical results overwritten.
Timing in scores/ covers newly executed calls only; metrics include the reused pilot windows.
Development-set results, not an independent test or student-model evaluation.
