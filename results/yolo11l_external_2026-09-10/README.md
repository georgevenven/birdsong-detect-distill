# Released YOLO11l comparison

Add BirdBox's released large checkpoint alongside nano; no new training or SongMAE runs.
Keep Qwen manually paused. Historical tables, figures, checkpoints and reports are unchanged.

Use the identical manifest from `results/current_external_2026-09-09`: calibrate separate
pixel/common-band/temporal IoU thresholds on 41 Powdermill segments from Recordings 2–4,
then freeze them before complete Recording_1 reporting and external evaluation. No test tuning.
Report WABAD (4,264 recordings / 68 sites), Hawaii (635), XC-AJ (288), and NIPS4Bplus (674).
Preserve all existing exclusions, ignored intervals, reference masks and aggregation rules.

Native released-model input: first audio channel, original sampling rate, linear 500–12,000 Hz
spectrogram, magma, six-second windows with five-second stride, 1024-pixel inference size.
The existing renderer matches the pinned BirdBox release. FP32, batch 8, candidate floor
0.00001, native NMS IoU 0.7, max_det 10,000 (fail if reached). Retain all predicted boxes.
Shared scoring only: 128 Slaney-mel bins over 20–16,000 Hz and 5-ms time bins; also save the
common-band diagnostic. AP uses continuous rasterized box confidences, maximum on overlap.
IoU uses the separately calibrated threshold. No Gaussian smoothing or morphological cleanup.

`run_manifest.json` pins code, released checkpoint, the evaluation manifest and prior results.
`source_before/` archives the three scripts extended for this run. The evaluator adds a
released-model branch without changing student inference or scoring. Each external recording
is checked against the existing nano result for audio hashes, duration and reference positives.
Shards are disjoint and merged only after coverage and calibration checks.

Run `.venv/bin/python scripts/run_released_yolo_large.py` through the dedicated user service.
Eight recording workers use four processes per GPU; model batch size stays fixed.
Results are resumable from per-recording scores. `tables/` contains expanded Tables 2/3,
CSV/TSV files, summary and source hashes; existing rows are reused without recomputation.
Existing SongMAE rows still describe the September 9 width-128 model, not later head ablations.
