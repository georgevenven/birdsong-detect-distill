# Matched external evaluation of the current students

**Complete.** [Tables 2 and 3](tables.md), full-precision CSV/TSV files and `summary.json` are ready.
[The final audit](audit.json) verified all 14 table-source reports, 11,876 score caches, 5,950 retained
prediction files and unchanged historical reports. Independent AP/count checks passed on 12 external
examples per current model, plus one YOLO Powdermill example. All evaluation services completed successfully.

Models: current SongMAE-Large with self-reviewed Qwen labels, and YOLO11n initialized from the released
BirdBox and retrained on exactly the same self-reviewed labels. Both use 10,000 s of XC training audio
(2,105 windows, 249 recordings) and 800 s of recording-disjoint XC validation audio (169 windows, 21 recordings).
`manifest.json` pins the split and the prior released-model reports that define external coverage.

SongMAE uses the completed hard-target BCE checkpoint from the September 9 stage ablation, selected by
minimum XC validation BCE over three epochs. Gaussian probability smoothing uses sigma 2 mel bins / 3 frames,
followed by one threshold, with no morphology. Native 5 s / 2.5 s stride inference is unchanged.
The Powdermill pixel scores reproduce the stage-ablation report exactly. Independently calibrated pixel
and frame thresholds both equal 0.06; this equality is a result, not a shared-threshold constraint.

YOLO retains its native box objective, released BirdBox initialization, 1024-pixel training size, batch 16,
seed 0 and 50-epoch budget. The checkpoint maximizes native box mAP50–95 on the separate XC validation set.
Epoch 32 was selected. Independently calibrated pixel and temporal thresholds both equal 0.01.
Images are the established 2048×512 viridis, 128-mel, five-second representation. Short windows are padded;
boxes are clipped to their owned training intervals. The native loader removes exact duplicate boxes,
which does not change the union foreground mask; no images are dropped.
Native inference retains confidence floor 0.00001, NMS IoU 0.7 and max_det 10,000 (hitting the cap fails).

Each model's pixel and temporal thresholds independently maximize mean segment IoU on the 41 Powdermill
segments from Recordings 2–4. Thresholds are written before complete Recording_1 reporting and then frozen
for all external datasets. AP uses continuous scores before thresholding; temporal scores take the maximum
across frequency after SongMAE smoothing. No threshold or checkpoint is selected using external results.

External coverage and aggregation are unchanged from `results/competitors_external_2026-09-08/`:
WABAD 4,264 primary recordings / 68 sites (site macro); Hawaii 635 recordings (recording macro);
XC-AJ 288 known-index-disjoint recordings; NIPS4Bplus 674 annotated clips, non-birds negative and Unknown
intervals ignored. All use 200-Hz scoring, with 128 Slaney-mel bins / 20–16,000 Hz for pixel masks.
The three inverted-label WABAD recordings remain excluded; zero-area rectangles contribute no positive area.
Every new recording is checked against the released baseline for source audio, scoring duration and reference positives.
Broader pretraining exposure is not fully certified. Matching labels does not isolate architecture from pretraining,
input representation, training objective or optimization.

Storage: resumable per-recording metric caches retain AP and every threshold's pixel/frame counts.
YOLO box predictions are retained. Dense SongMAE maps are retained only for three deterministic examples per
external dataset; all original Powdermill stage caches remain available. No historical files are deleted.

Run `bash scripts/run_current_external.sh songmae` or `qwen_yolo` with the intended `CUDA_VISIBLE_DEVICES`.
`CURRENT_SHARDS=4` partitions external recordings without changing native inference. Calibration is unsharded.
During WABAD evaluation, YOLO was increased to eight recording workers because rendering was CPU-bound.
All 328 completed score caches captured before the restart were checksum-verified unchanged; see
`worker_transition.json`. Model batch size, input representation, scoring and thresholds did not change.
After both finish, run `python scripts/summarize_current_external.py --results results/current_external_2026-09-09`.
It produces Tables 2/3 and full-precision CSV/TSV with precision/recall, reusing the released YOLO and BirdCODE
reports unchanged after source, coverage, reference and calibration verification. Historical figures are not updated here.
