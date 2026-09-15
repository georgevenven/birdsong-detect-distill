# Held-out XC: bare-linear SongMAE detector

Open 00_OVERVIEW.png first. Each individual PNG/PDF is square, 3.5 inches, with 600-dpi PNG output.
Top: clean spectrogram, NOT human ground truth. Bottom: cyan contours of the actual binary prediction.
Frozen SongMAE-Large + Linear(768,32), 24,608 head parameters; seed 0, 2,000 s training. Threshold 0.07 selected on Powdermill Recordings 2–4, never on these XC examples.
Post-processing: sigmoid probabilities, Gaussian smoothing sigma=2 mel bins and 3 frames (15 ms), then one threshold. No hysteresis, closing, component removal, hole filling or cosmetic contour smoothing.
Full-source 5-second inference windows overlap by 2.5 seconds and combine by maximum. Smooth before cropping the displayed five-second excerpt. Native dequantized XC spectrograms use the backbone training normalization.
seed-17 shuffled XCL validation recordings lasting 10–120 s; distinct focal species/geographic cells; random aligned 5 s excerpt before inference.
disjoint from entire XCL training index, detector train/validation IDs, 50k teacher queue and XC-AJ; source-ID audit, not cross-ID acoustic deduplication.
These are qualitative examples, not a localization benchmark. Focal species metadata does not verify the species inside an excerpt.

## Recording attribution

- 01_XC339719_55-60s: Bram Piot; https://xeno-canto.org/339719; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 1281; coordinates 12.4674, -16.787.
- 02_XC303542_35-40s: Qin Huang; https://xeno-canto.org/303542; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 8626; coordinates 5.873, 117.9399.
- 03_XC819373_15-20s: Guillermo Menéndez (gmmv80); https://xeno-canto.org/819373; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 2185; coordinates -24.9671, -57.2932.
- 04_XC617124_0-5s: Robert Ekman; https://xeno-canto.org/617124; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 7088; coordinates 59.9674, 17.9371.
- 05_XC624104_0-5s: Chris Batty; https://xeno-canto.org/624104; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 8342; coordinates 53.9299, -2.9833.
- 06_XC252675_0-5s: Alex Rinkert; https://xeno-canto.org/252675; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 6407; coordinates 36.9526, -122.0599.
- 07_XC410338_15-20s: Rolf A. de By; https://xeno-canto.org/410338; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 1256; coordinates 9.868, 38.917.
- 08_XC589136_5-10s: Wonseok Jang; https://xeno-canto.org/589136; license https://creativecommons.org/licenses/by-nc-sa/4.0/; focal species code 7778; coordinates 35.9334, 126.7535.
