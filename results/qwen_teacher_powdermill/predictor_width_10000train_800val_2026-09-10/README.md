# Backbone-matched predictor width

One training seed (0), best checkpoint within five epochs by minimum XC validation BCE.
Frozen SongMAE backbones; existing one-self-review Qwen labels; exactly 10,000 s XC
training audio and the same separate, recording-disjoint 800 s XC validation set.
Evaluate only Powdermill: Recordings_2–4 calibrate each IoU threshold; complete Recording_1
supplies reporting scores. Keep hard-mask BCE, AdamW lr 0.001, weight decay 0.0001,
batch 16, dropout 0.1, probability smoothing (2 mel bins, 15 ms), and no morphology.

| Backbone | Original predictor d | Matched predictor d | Matched trainable parameters |
| --- | ---: | ---: | ---: |
| Micro | 128 | 128 | 281,888 |
| Base | 128 | 384 | 1,730,336 |
| Large | 128 | 768 | 6,114,848 |

Match latent dimension, not backbone depth or total parameter count. Preserve the learned
input projection, positional embeddings, one four-head transformer layer, feedforward
width 2*d, and per-patch output. No backbone finetuning, learning-rate changes or extra Qwen
review calls. Same seed/data membership; different head dimensions consume RNG differently.

Matched d=128 controls also use best-of-five selection. Reuse Micro's epoch-4 and Large's
epoch-3 checkpoints from the completed 15-epoch runs: each is also the minimum within
the first five epochs under the unchanged constant-learning-rate schedule. Base's best
of five is epoch 4, which was not saved, so retrain its d=128 control for five epochs.
Three new runs: Base d128, Base d384, Large d768. Micro is unchanged and shared between
conditions. A real 16-window Large d768 forward/backward/optimizer memory check passed.

`manifest.json` pins data, references, backbone revisions through source reports, and code.
Outputs: `comparison.csv`, `learning_curves.csv`, `summary.json`, `caption.txt`, and square
`predictor_width_ap.{png,pdf,svg}`. Retain three dense prediction examples per fresh run,
plus exact scores and probability hashes for every evaluated segment. Historical figures
remain untouched. Qwen stays manually paused; this runner cannot restart it.
