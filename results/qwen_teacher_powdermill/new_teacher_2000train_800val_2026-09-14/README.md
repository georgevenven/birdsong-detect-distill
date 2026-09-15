# New teacher labels: 2,000-second head comparison

Frozen SongMAE-Large, width 128: the existing zero-attention pointwise head versus one self-attention layer (not cross-attention). Three paired seeds (0, 1, 2), five epochs, best checkpoint by recording-disjoint XC validation BCE. Hard binary teacher masks, AdamW 0.001, batch 16, no target smoothing or TV.

Training: 400 five-second windows, 400 source recordings and 400 focal species. Validation: 160 additional windows/recordings, 800 seconds. Both use the new XC pipeline: reasoning with coordinate axes, then one numbered self-review. Confidence is retained in annotations but not used as a BCE target.

`snapshot.json` freezes the completed-window pool before training. A seed-17 shuffle selects training and validation without consulting labels or Powdermill results. This is a preliminary available-pool comparison; annotation latency can affect which windows have completed. Input spectra, raw-to-mapped boxes, exclusion IDs, disjoint recordings and exact durations were checked. Selected annotations and initial parents are hash-protected while V100 continues appending other windows.

Evaluation uses all Powdermill intervals: calibrate each model's single threshold on original Recordings 2–4; report original Recording 1. Smooth probabilities with Gaussian sigma (2 mel bins, 3 frames); pixel AP uses continuous smoothed scores. No morphology, no held-out external-dataset evaluation. Seed SD describes initialization variability, not dataset uncertainty.

## Detached operation

Unit: `birdsong-detector-newteacher2k-20260914.service`.

- Driver log: `driver.log`; per-condition logs: `logs/`.
- Progress: `status.json`, `gpu0.json`, `gpu1.json`.
- Final summary: `pointwise/summary.json`, `pointwise/comparison.csv`.
- Checkpoints and three cached example predictions per model: `/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/new_teacher_2000train_800val_2026-09-14/`.

Twins' XC client drained gracefully and its local Qwen server was stopped. V100's client, server and relay tunnel were left running. The detector runner does not restart or stop any Qwen service; Twins remains paused afterward. Closing the terminal or Codex does not stop this user service (linger enabled).

To resume Twins annotation later, explicitly start `birdsong-qwen-xc50k-twins-20260914.service` after detector GPU work is finished; its existing dependency starts the local Qwen server. Saved initial calls and self-reviews are reused.
