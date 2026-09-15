# External competitor evaluation

**Complete.** All ten competitor reports passed cross-model coverage/calibration checks. All 12,690 native caches passed checksums; independent scikit-learn AP and direct threshold counts/IoU matched on 27 sample recordings. Paper-ready CSV/TSV/Markdown tables and `summary.json` are available here. See `audit.json` for details.

Run the released BirdBox YOLO11n and the fixed teacher-trained YOLO11n on WABAD/Hawaii (pixel metrics), and those two models plus BirdCODE on XC-AJ/NIPS4Bplus (frame metrics). **No external SongMAE evaluation is authorized in this experiment.** Its table cells remain pending detector development.

Native input, checkpoint and prediction settings are unchanged from `results/baselines_matched_2026-09-07/powdermill_final/`. Each final report enforces identical inference metadata and the same coverage-manifest hash as its calibration report. The two YOLO models retain confidence floor 0.00001, NMS IoU 0.7 and maximum 10,000 detections; hitting that cap fails the run. No event-box matching metrics are used in these tables.

| Competitor | Pixel threshold | Frame threshold |
|---|---:|---:|
| Released YOLO11n | 0.05 | 0.04 |
| Teacher-trained YOLO11n | 0.01 | 0.01 |
| BirdCODE | Not applicable | 0.20 |

Thresholds were selected on the 41 Powdermill calibration segments from original Recordings 2–4, and are frozen before external evaluation. No source-recording species list or geography oracle is supplied to any model. No retraining of the competitors is performed.

## Coverage and metrics

- WABAD: infer all 4,267 local recordings across 68 strongly annotated sites, including empty recordings. The annotation audit found 44 nonpositive-area boxes in 13 recordings. Three recordings contain inverted bounds and are excluded uniformly from primary scoring: `CB/CB_20180524_061004.wav`, `CB/CB_20180524_061005.wav`, and `SPMCO/SPMCO_20210416_102000.wav` (the latter has 32 reversed time intervals). This leaves 4,264 primary recordings across all 68 sites. Ten zero-extent boxes in other recordings contribute zero area before rasterization; seven negative onsets are clipped at the recording boundary. Native predictions and raw labels are preserved, and an uncorrected all-recording diagnostic summary is retained. Primary AP and IoU are site-macro: recording means within each site, then equal-weight means across sites. Precision/recall pool pixel counts within sites, then average equally across sites. Recording-weighted summaries and per-site results are also retained.
- Hawaii: all 635 original Zenodo FLAC files, 50.883 hours across four sites. All 59,583 original rows are preserved on disk; the 196 zero-duration boxes contribute zero reference area, leaving 59,387 positive-area rectangles. Do not expand zero-duration labels into artificial frame positives. Primary AP/IoU are recording means, and precision/recall pool pixel counts across recordings; supplementary site-macro results are retained.
- XC-AJ (XCSL): retain the already-frozen 288-recording known-index-disjoint subset of 967 recordings. The remaining 679 overlap known XCL pretraining/model-selection IDs; this does not certify absence of duplicates under other IDs or undocumented exposure.
- NIPS4Bplus: all 674 locally available strongly annotated clips. Bird calls are positive, known non-bird classes negative, and Unknown intervals ignored.

All competitors are scored on the same 200-Hz time grid. Pixel metrics use 128 Slaney-mel bins spanning 20–16,000 Hz; rectangles are unioned into reference masks, and overlapping predicted boxes take maximum confidence. Full-band scores are primary. A supplementary common-band score restricts only the scoring rows, never the native model input. These are mel-weighted, box-derived area masks, not hand-drawn spectrogram segmentations.

Frame metrics collapse frequency or species using maximum score for any-bird occupancy. AP is exact noninterpolated foreground AP of continuous scores, averaged over positive recordings; IoU averages all recordings with empty union = 1. Temporal precision/recall pool frame counts. AP is **not** bounding-box AP@0.5, and temporal IoU is **not** event IoU matching.

Hawaii was an external evaluation collection for the released BirdBox, but its annotations can omit faint/unidentifiable birds. Broad pretraining provenance is not fully certified. See the [BirdBox paper](https://arxiv.org/html/2606.10407v1) and the [original Hawaii release](https://zenodo.org/records/7078499). These adapted class-agnostic metrics are not directly comparable to the papers' species-aware or box-matching metrics.

## GPU coordination

The Qwen annotation client was frozen in place; its server drained to zero active requests before being stopped normally. At pause, 2,985 completed windows and 1,284 stage-checkpoint files were backed up under `artifacts/competitors_external_2026-09-08/annotation_pause_snapshot/`. Annotation prefix SHA-256: `061f72ee6ce263b8e4598e159d62f6ebd48c494db4f15aba3d6d9aa1ef6f6ab9`.

GPU 0 ran released YOLO. GPU 1 first refreshed the Powdermill figure models, then BirdCODE and teacher-trained YOLO. Each YOLO dataset initially used four disjoint recording shards on its GPU, retaining the same native batch size and preprocessing. BirdCODE used one worker to preserve its memory budget. The initial serial released-YOLO run was interrupted to enable this parallelism; all 183 completed prediction caches passed their checksums and were reused.

Qwen's server was restored and confirmed healthy before the same annotation client was thawed at 15:17:38 PDT. All 1,284 saved checkpoint files / 6,072 saved stages and the exact 45,968,955-byte annotation prefix were preserved. At 15:18:11, 32 requests were active and six fresh windows had completed (2,991 total). One transient empty-response parse error occurred after thaw; saved stages remained intact, and the existing wrapper retries incomplete/error windows on its next pass.

The shorter YOLO temporal datasets also ran with two workers per GPU alongside the disjoint long soundscape files. Their finished reports are reused by the main runner. All native model batch sizes and thresholds remain unchanged.

Teacher-trained YOLO's Hawaii evaluation was expanded from four to eight recording workers (`COMPETITOR_SHARDS=8`) to use spare capacity. All 139 completed caches were checksum-verified and reused; no incomplete caches were found. The first four-worker logs are preserved under `logs/hawaii_4worker_initial/`. Only worker partitioning changed, not native inference, batch size, scoring or calibration.

The original released-YOLO wrapper exited 127 after writing the completed Hawaii report. A fresh invocation verified/reused all four final reports and exited 0 without additional inference. All expected recording counts passed, and independently recomputed Hawaii sample AP/counts matched. The nonzero wrapper exit is retained in the service journal; no report was replaced.

## Reproduction

`bash scripts/run_external_competitors.sh MODEL`, with `MODEL` one of `birdbox`, `qwen_yolo`, `birdcode`, uses the selected `CUDA_VISIBLE_DEVICES` GPU. Results and logs are written here; native prediction caches are under `artifacts/competitors_external_2026-09-08/MODEL/`. The runner verifies and reuses finished reports and hash-checked prediction caches. Sharded reports are marked diagnostic until `scripts/merge_external_shards.py` verifies that every eligible recording appears exactly once and combines their per-recording metrics. No thresholds are selected during merging.

`scripts/summarize_competitor_tables.py` validates all ten final reports against frozen calibration and checks cross-model recording/audio/reference coverage before exporting CSV, TSV and Markdown tables. `summary.json` retains lightweight aggregate results and provenance; `table_sources.json` records full-report hashes. Large per-recording reports, shard files and prediction caches stay on disk but are Git-ignored. Nothing is deleted.
