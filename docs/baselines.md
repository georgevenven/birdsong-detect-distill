# Baseline evaluation

Use native model inputs and one shared scorer. Earlier files in `results/reference/` and
`results/reproduced/` are historical diagnostics, not the corrected paper comparison.
No existing checkpoints or figures are replaced.

The next external competitor run is `results/competitors_external_2026-09-08/`: released and
teacher-trained YOLO on WABAD/Hawaii/XC-AJ/NIPS4Bplus, BirdCODE on XC-AJ/NIPS4Bplus, and no
external SongMAE inference. Four disjoint recording workers per YOLO preserve the native
inputs and batch sizes. `merge_external_shards.py` verifies coverage before producing a final
report. The annotation audit excludes three WABAD recordings with inverted bounds uniformly
from primary scoring (4,264 retained), and gives zero-extent reference boxes zero area before
rounding. Uncorrected all-recording diagnostics and native caches are preserved. See that
run's README for exact exclusions and metric aggregation.

The new SongMAE Powdermill figures are separately under
`results/qwen_teacher_powdermill/hard_bce_figures_2026-09-08/`: hard targets, plain BCE,
probability-space Gaussian smoothing and AP-only quantitative panels. Earlier raw-score and
soft-target/TV experiments below remain historical results, not the updated detector recipe.

## Native inputs

| Model | Input | Inference |
|---|---|---|
| Released BirdBox YOLO11n | Native sample rate, first channel, linear 500–12,000 Hz, magma, approximately 1024-square | 6 s / 5 s stride; confidence floor 0.00001; NMS IoU 0.7; max 10,000 detections (fail if reached) |
| YOLO11n trained on our labels | Mono 32 kHz, 128 Slaney-mel bins, FFT 1024 / hop 160, 20–16,000 Hz, peak-relative dB, viridis 2048×512 | 5 s / 2.5 s stride; same confidence floor and NMS as released YOLO |
| SongMAE-Large | Native SongMAE mel preprocessing and frozen backbone | Existing 5 s / 2.5 s maximum-overlap inference; raw probabilities, no cleanup |
| BirdCODE | Mono float32 32 kHz; official frontend: HTK 256 mel bins, FFT 2048 / hop 256, power dB with top_db=80, mean −13.369 / std 13.162 | Official `FrameDetector.run`, 5 s / 2.5 s center-retained inference; 7.6 Hz / 9,422 species; batch size 4 |

BirdBox is pinned to [094d5c3](https://github.com/org-arl/birdwatch-public/tree/094d5c36abda3d7f29151a2dca6d00282cbfbc8d).
The Python renderer includes DSP.jl's fast FFT padding, symmetric Hann window, frequency
boundary rounding, interpolated magma colors and rounded RGB bytes. Runtime comparisons
against the original Julia renderer (DSP 0.8.6) were pixel-identical at 32, 44.1 and 48 kHz.
The released YOLO11n checkpoint SHA-256 is checked before use. This is the nano baseline,
not a claim to represent the strongest YOLO variant; `--variant yolo11l` supports the other release.

BirdCODE [code 617c640](https://github.com/earthspecies/sound-event-detection/tree/617c6408fdd1b56c3873ef2d17006ffbaee61783)
and [weights b7a8ee2](https://huggingface.co/EarthSpeciesProject/sed-birdcode/tree/b7a8ee261579278f44f636baf4024ce384ceb490)
are pinned. Resampling follows its loader: channel mean, `kaiser_best`, `scale=True`.
Input is explicitly zero-padded to its existing final-window boundary, then scored only
over the original duration. This preserves the last partially valid coarse frame that
`run()` otherwise rounds away. Model windows and their predictions are unchanged.
We retain all output bird species; no ground-truth species list or geography oracle is supplied.

## Matched training and splits

`scripts/make_benchmark_manifest.py` freezes the teacher-file hash, training recording IDs,
Powdermill reporting intervals and known-pretraining exclusions. The current manifest is
`results/baselines_matched_2026-09-07/manifest.json`:

- Exactly 10,000 s, 2,103 windows, 266 XC recordings from self-review labels for both students.
- YOLO uses the same integer time/mel target rectangles as SongMAE. Shortened tiles are
  padded to five seconds, not stretched; labels are clipped and placed on that fixed grid.
- YOLO is initialized from the released BirdBox checkpoint and trained for a fixed 50 epochs.
  All 2,103 windows train the model; no validation fraction is removed. The final checkpoint,
  not a best-on-training or best-on-reporting checkpoint, is used. Ultralytics' mandatory
  validation path aliases training for diagnostic output only; those scores are not held-out.
  SongMAE retains its existing three-epoch training. Equal labels do not mean equal optimization
  budgets or pretraining; this is not an isolated architecture-only causal comparison.
- Powdermill: calibrate on all 41 segments from original Recordings 2–4; report on the same
  30 Recording_1 segments / 8,235 s scored in the teacher/student table.
- Powdermill is development data for both this work and BirdCODE, never an unseen test set.
- XC-AJ (called XCSL in BirdCODE) contains 967 recordings. The union of known XCL training
  and validation indices excludes 679, leaving 288. The teacher training subset overlaps none.
  This is a conservative known-index-disjoint subset, not proof of complete pretraining
  provenance or absence of duplicate audio under other IDs. Do not call the full 967 held out.
- WABAD: all 4,267 public recordings across the 68 strongly annotated sites, including 31
  empty clips; primary aggregation is equal-weight site macro. Never select just one clip/site.
- NIPS4Bplus: only the 674 locally available strongly annotated clips; birds are positive,
  insects/amphibians/humans/noise negative, Unknown intervals ignored. No frequency score is
  computed from its temporal-only labels. Broader pretraining provenance remains unverified.

## Metrics

The primary area metrics use the same 200-Hz temporal grid and 128-bin Slaney-mel frequency
grid for every model. Human rectangles are unioned into reference masks; these are **box-derived
area labels, not hand-drawn pixel segmentation**. YOLO boxes are converted from seconds/Hz
to that grid and rasterized with their confidence; overlaps take the maximum.

Report exact noninterpolated foreground mask AP and mean segment IoU for SongMAE/YOLO,
with full-band scores primary and rows intersecting 500–12,000 Hz as a supplementary comparison.
The band restriction is applied only to predictions and references during scoring: SongMAE
still receives its unchanged full-band input, without cropping or rescaling. This is
mel-weighted area, not Hz·seconds.
BirdCODE is temporal-only: maximum probability across all species gives any-bird occupancy.
For every model, also report temporal occupancy AP/IoU on the frequency union. AP excludes
segments without positive reference area; empty-union IoU is one. Counts are retained.

Select separate full-band, common-band and temporal thresholds by development-mean IoU;
never select them on reporting/test intervals. AP uses unthresholded scores, not cleaned
binary masks. Existing post-processed qualitative figures are illustrations and are not the
raw scoring pipeline. Do not silently change the numerical pipeline to match a figure.

Optional `--event-metrics` preserves native event structure:

- BirdCODE: threshold each species, extract events, merge same-species gaps <1 s, remove
  durations <0.01 s, temporal NMS at 0.8, then discard species identity for matching.
- YOLO: native boxes plus cross-window time-frequency NMS at 0.7; frequency collapse does
  not merge distinct surviving boxes. Area metrics use all native boxes, without this extra NMS.
- SongMAE: eight-connected regions at each threshold, at least 32 pixels and three frames;
  convert each surviving region separately to an onset/offset pair. No global one-second merge.
- Match class-agnostically by maximum cardinality at temporal IoU >0.2 / >0.5. Sweep 101
  thresholds and integrate the interpolated PR envelope, explicitly distinct from exact area AP.
  Event F1 operating points maximize pooled development F1. No-positive event AP is undefined.
  At coverage boundaries both GT and predictions are identically clipped/split, without
  concatenating disconnected intervals into a continuous recording.

These adapted detection-only metrics are not directly comparable to published species-aware
BirdCODE mAP or BirdBox's IoMin/duplicate-tolerant box metrics.

## Run

The ordinary `.venv` handles SongMAE and YOLO. BirdCODE dependencies are isolated:

```bash
python -m venv --system-site-packages .venv-birdcode
.venv-birdcode/bin/python -m pip install -r requirements/birdcode.txt
# Match this installation's PyTorch ABI; choose the corresponding CPU/CUDA build elsewhere.
.venv-birdcode/bin/python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
.venv-birdcode/bin/python -m pip install -e . --no-deps

bash scripts/run_matched_baselines.sh
```

The runner uses only the selected visible GPU (default physical GPU 1), evaluates serially,
and writes new prediction caches and reports. For native-event metrics, use a separate report
folder: `EVENT_METRICS=1 BENCHMARK_RESULTS=results/baselines_matched_events bash scripts/run_matched_baselines.sh`.
Prediction caches are reusable; `--score-only` does not load a GPU model.
Cache reuse checks code/configuration, audio, references and prediction hashes. Diagnostic
`--maximum` runs cannot be used as calibration for a final evaluation.

External evaluation requires this same model's complete Powdermill report via `--calibration`.
Use `scripts/evaluate_baselines.py --help` for all options. For NIPS4Bplus, `--root` points at
`data/nips4bplus`; for the other datasets it points at the BirdCODE raw archive directory.
The manifest excludes known overlapping XC IDs automatically. It does not silently replace
the full XCSL benchmark or imply identical dataset coverage to a published table.

The initial 0.0001 floor was measurably sensitive on development data, so final reports use
0.00001 for both YOLO models. The first segment of each calibration source was checked at
0.0001, 0.00001 and 0.000001, without using reporting data. This sampled check is diagnostic,
not a proof of convergence on every recording. Final reports are in `powdermill_final/`;
the earlier `powdermill/` reports are preserved. A detection-cap hit is a hard failure.
`scripts/check_yolo_confidence_floor.py` performs a paired sensitivity check, defaulting to
the first segment of each calibration source; `--all-development` expands it to all 41.

Sources: [BirdBox paper](https://arxiv.org/html/2606.10407v1),
[BirdCODE paper](papers/BirdCODE.pdf), and the pinned official repositories above.
