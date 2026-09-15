# Matched 250-window teacher pilot

| Condition | Qwen pixel AP | Flash Vision Exp pixel AP | Qwen 2D IoU | Flash Vision Exp 2D IoU |
| --- | ---: | ---: | ---: | ---: |
| direct_plain | 0.2297 | 0.3697 | 0.1296 | 0.3017 |
| direct_axes | 0.3575 | 0.4022 | 0.2689 | 0.3435 |
| reasoning | 0.4900 | 0.4009 | 0.4221 | 0.3327 |
| self_review_1 | 0.4967 | 0.3911 | 0.4285 | 0.3231 |
| self_review_2 | 0.4981 | 0.3981 | 0.4307 | 0.3258 |

Report: identical 150 Recording_1 windows; thresholds independently calibrated on the same 100 Recording_2–4 windows.
Axes frozen from Qwen for both reasoning chains. AP uses continuous scores without smoothing.
Zen model is deepseek-v4-flash-vision-exp; V4.1 identity is not established. Provider generation controls differ.
Pilot teacher results only, not full Powdermill or student evaluation.
