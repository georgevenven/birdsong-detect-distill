# XC-validation threshold calibration

Counterfactual threshold-only evaluation of the same fully fine-tuned Micro, Base, and Large checkpoints trained on 3,205 seconds (seed 0). No retraining, changed checkpoints, new Powdermill inference, or gallery/default changes.

Select each model's threshold by maximum mean pixel-mask IoU across all 160 original XC validation windows (800 seconds), using the existing 0–1 grid in 0.01 increments and choosing the lowest threshold on ties. Ten windows have empty teacher masks; all 160 contribute to IoU. These are Qwen reasoning + one self-review labels, not human references. This validation set previously selected training epochs too.

Use native source spectrograms with the model's training normalization, full-source five-second windows / 2.5-second stride, maximum overlap aggregation, and Gaussian probability smoothing with sigma (2 mel bins, 3 frames). Only annotated five-second windows contribute calibration pixels; unlabeled surrounding context is never treated as negative. Input, reference-mask and interval hashes match across all three models.

Freeze XC thresholds before applying them to the saved Powdermill confusion-count curves. Powdermill scoring covers all 77 segments / four original recordings (23,100 seconds). AP averages the 76 reference-positive segments; IoU averages all 77. The prior comparator used leave-one-original-recording-out Powdermill threshold calibration, not one threshold tuned on all four.

| Model | XC threshold | Pixel AP (unchanged) | Prior Powdermill-calibrated mean IoU | XC-calibrated mean IoU | XC-calibrated pooled precision | XC-calibrated pooled recall |
|---|---:|---:|---:|---:|---:|---:|
| Micro | 0.26 | 0.686493 | 0.461263 | 0.353634 | 0.757653 | 0.375212 |
| Base | 0.29 | 0.757161 | 0.488900 | 0.403735 | 0.835302 | 0.394174 |
| Large | 0.36 | 0.767669 | 0.520032 | 0.308482 | 0.915167 | 0.283295 |

The XC-selected operating points transfer poorly to human-labeled Powdermill: precision increases but recall falls. This experiment does not distinguish recording-domain shift from differences in teacher versus human annotation policy. Threshold source does not alter continuous-score pixel AP. Using XC for thresholds does not retroactively make Powdermill an untouched test set; previous model and postprocessing development used Powdermill.

Reproduce: `scripts/calibrate_finetuned_on_xc.py --sizes micro base large` with a free visible GPU and the existing environment. Per-model `manifest.json`, `calibration.json`, and `comparison.json` retain exact results; `windows/` stores source hashes and calibration curves. Cropped continuous predictions are cached under `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/xc_threshold_calibration_full_encoder_3205s_2026-09-14/`.
