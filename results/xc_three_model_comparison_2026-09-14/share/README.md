# Three-model held-out XC comparisons

Open 00_OVERVIEW.png first: eight excerpt rows, three model columns. Individual PNG/PDFs retain clean input above each prediction.
Left: frozen SongMAE-Large + linear (threshold 0.07). Middle: frozen SongMAE-Large + one 128-wide transformer layer (0.05). Right: fully fine-tuned SongMAE-Large encoder + linear (0.09).
All models: seed 0, the same 2,000 s XC training / 800 s validation split, checkpoint selection by XC validation BCE.
Exact original source recordings, excerpt times, spectrograms and cached post-processed masks. No inference or training rerun. Model-specific thresholds were selected on Powdermill Recordings 2–4, not these examples.
Same full-source probability smoothing sigma (2 mel,3 frames); original per-model Powdermill Recordings_2–4 thresholds retained.
Clean inputs are not human ground-truth annotations. Cyan contours are model predictions.
Standard Matplotlib rendering. Cyan outlines follow the existing binary masks; no additional smoothing, morphology, contour cleanup or normalization changes.

## Recording attribution

- 01_XC339719_55-60s: Bram Piot; https://xeno-canto.org/339719; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 1281; coordinates 12.4674, -16.787.
- 02_XC303542_35-40s: Qin Huang; https://xeno-canto.org/303542; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 8626; coordinates 5.873, 117.9399.
- 03_XC819373_15-20s: Guillermo Menéndez (gmmv80); https://xeno-canto.org/819373; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 2185; coordinates -24.9671, -57.2932.
- 04_XC617124_0-5s: Robert Ekman; https://xeno-canto.org/617124; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 7088; coordinates 59.9674, 17.9371.
- 05_XC624104_0-5s: Chris Batty; https://xeno-canto.org/624104; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 8342; coordinates 53.9299, -2.9833.
- 06_XC252675_0-5s: Alex Rinkert; https://xeno-canto.org/252675; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 6407; coordinates 36.9526, -122.0599.
- 07_XC410338_15-20s: Rolf A. de By; https://xeno-canto.org/410338; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 1256; coordinates 9.868, 38.917.
- 08_XC589136_5-10s: Wonseok Jang; https://xeno-canto.org/589136; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 7778; coordinates 35.9334, 126.7535.

