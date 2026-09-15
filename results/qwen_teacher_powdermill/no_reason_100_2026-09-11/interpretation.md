# Preliminary result and calibration caveat

All 100 annotations completed on the first attempt: no truncated outputs or
returned reasoning. The server rendered an empty, closed thinking prefix.
Generation took 405.990 seconds, excluding server startup: 53,240 output tokens,
532.4 per window, and 131.14 aggregate output tokens/s. Mean request latency was
99.60 seconds; throughput was one completed window per 4.06 seconds on average.
The dedicated server stopped successfully. The original XC queue remains paused.

Evaluation uses 60 identical windows (300 seconds) from Recording 1. The other
40 windows (200 seconds) are reserved for threshold calibration. This is a small,
checkpoint-available development subset, not full-Powdermill or final-test evidence.

| Teacher | Pixel AP | IoU at shared threshold 0.01 | Pixel precision | Pixel recall |
| --- | ---: | ---: | ---: | ---: |
| Historical `direct` (reasoning mode unverified) | 0.4394 | 0.3237 | 0.7776 | 0.4018 |
| New explicitly disabled-reasoning calls | 0.4668 | 0.3713 | 0.8378 | 0.4235 |

The shared-threshold results above are supplementary diagnostics, not the original
per-model calibration outcome. They use `summarize(evaluation_rows, {'full': 1})`
from `comparison.json`, where index 1 is threshold 0.01. AP is threshold-free.

Under the frozen 0.00–1.00 threshold grid, historical direct selected 0.00:
all pixels are positive because the existing mask rule is score >= threshold.
On calibration, that gave mean IoU 0.2393, exceeding 0.2158 at threshold 0.01.
Its resulting evaluation IoU is therefore 0.1879. New no-reasoning predictions
selected 0.01, giving calibration IoU 0.2875 and evaluation IoU 0.3713.
Those original outcomes remain unchanged in `comparison.json` and `comparison.md`.
The apparent calibrated IoU improvement is magnified by this degenerate operating
point; it should not be interpreted as a near-doubling in localization accuracy.

Pixel AP is 0.0274 higher for the new annotations on this matched sample. This is
encouraging preliminary evidence, but neither a significance claim nor proof
that disabling reasoning helps: the historical comparator's actual reasoning
mode was not recorded reliably. No student was trained in this pilot.
