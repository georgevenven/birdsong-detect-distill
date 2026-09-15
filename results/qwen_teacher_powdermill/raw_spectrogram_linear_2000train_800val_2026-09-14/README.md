# Linear detection directly on spectrograms

Normalized log-mel column (128 bins) -> Linear(128,128) -> 128 foreground logits at each 5 ms; no encoder weights, hidden activation, learned positions or temporal mixing.
Same 2000 s training/800 s validation source splits, hard-mask BCE, optimizer, five epochs and best XC validation selection as seed-0 SongMAE control. Seed is shared; exact initialization or minibatch ordering is not asserted across different architectures.

| Input features (seed 0) | Pixel AP | 2D IoU |
|---|---:|---:|
| Raw 128-bin spectrogram | 0.511 | 0.245 |
| Frozen SongMAE-Large latents | 0.761 | 0.509 |

All 77 five-minute segments (6 h 25 min), leave-one-original-recording-out threshold calibration. Same probability smoothing; segment-macro metrics. No external dataset evaluation.
Selected epoch: 5 of 5, by XC validation BCE. Raw probe: 16,512 parameters; SongMAE probe: 24,608 plus frozen backbone.
Single-seed exploratory baseline. Different input dimension, parameter count and receptive field; not an isolated test of pretraining versus random encoder features.
Existing website, figures, benchmark results and annotation services were not changed.
