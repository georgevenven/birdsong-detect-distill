# XC comparison: common threshold 0.09

Open 00_OVERVIEW.png first. Eight individual three-column PNG/PDF comparisons also include clean inputs above predictions.
Left: frozen encoder + linear. Middle: frozen encoder + transformer head. Right: fully fine-tuned encoder + linear. All use threshold 0.09.
Same three models and eight excerpts as the previous grid. This does not include the newer transformer-block-only fine-tuning variant.
Previous thresholds were 0.07, 0.05 and 0.09 respectively. Only the first two models therefore change; full-fine-tuning masks must be identical.
Same cached continuous probabilities, source spectrograms, full-source smoothing and display. No retraining, inference or smoothing rerun; binary masks are recomputed at the shared threshold.
Original full-source probability Gaussian sigma (2 mel,3 frames) retained; float64 comparison against common threshold 0.09. No additional smoothing or morphology.
A shared numeric threshold does not guarantee matched precision, recall or score calibration. Qualitative operating-point check only; original calibrated evaluations remain unchanged.
Clean inputs are not human ground-truth annotations. Cyan contours are model predictions.
threshold_effect.json records how many foreground pixels disappeared; these counts are not accuracy measurements.

## Recording attribution

- 01_XC339719_55-60s: Bram Piot; https://xeno-canto.org/339719; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 1281; coordinates 12.4674, -16.787.
- 02_XC303542_35-40s: Qin Huang; https://xeno-canto.org/303542; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 8626; coordinates 5.873, 117.9399.
- 03_XC819373_15-20s: Guillermo Menéndez (gmmv80); https://xeno-canto.org/819373; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 2185; coordinates -24.9671, -57.2932.
- 04_XC617124_0-5s: Robert Ekman; https://xeno-canto.org/617124; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 7088; coordinates 59.9674, 17.9371.
- 05_XC624104_0-5s: Chris Batty; https://xeno-canto.org/624104; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 8342; coordinates 53.9299, -2.9833.
- 06_XC252675_0-5s: Alex Rinkert; https://xeno-canto.org/252675; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 6407; coordinates 36.9526, -122.0599.
- 07_XC410338_15-20s: Rolf A. de By; https://xeno-canto.org/410338; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 1256; coordinates 9.868, 38.917.
- 08_XC589136_5-10s: Wonseok Jang; https://xeno-canto.org/589136; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 7778; coordinates 35.9334, 126.7535.

