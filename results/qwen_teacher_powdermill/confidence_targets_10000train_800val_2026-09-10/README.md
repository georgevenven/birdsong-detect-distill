# Qwen-confidence BCE targets

Train one frozen SongMAE-Large + one-layer d384 predictor with fractional BCE targets.
For each pixel, y is the maximum confidence of covering target_vocalization,
uncertain_vocalization, or chorus boxes, or zero outside all foreground boxes.
Minimize -y log(p) - (1-y) log(1-p), not confidence-weighted hard-label loss.
All valid pixels contribute, including uncertain regions. No spatial target smoothing,
total-variation penalty, confidence cutoff, or ignored uncertainty is used.

Reuse the completed hard-mask Large+d384 baseline. Match seed 0, initial head weights,
minibatch order, all 10,000 s XC training audio (2,105 windows / 249 recordings), and the
recording-disjoint 800 s XC validation set (169 windows / 21 recordings). Train five
epochs and select minimum validation BCE against the corresponding target policy.
AdamW lr 0.001, weight decay 0.0001, batch 16, dropout 0.1; backbone stays frozen.
Validation BCE uses different target values across policies and is not directly comparable.
Qwen's self-reported label confidences are not calibrated probabilities of bird presence.

The training foreground covers 22,591,551 of 256,000,000 valid pixels, unchanged from
hard masks; its mean confidence target is 0.683947 after maximum-overlap rasterization.
Validation foreground covers 2,622,385 of 20,480,000 valid pixels, with mean target
0.684365. No labels, audio windows, or human reference pixels are removed.

Powdermill only: Gaussian probability smoothing (2 mel bins / 15 ms), one threshold
maximizing mean 2D IoU on Recordings_2–4 (41 segments), and complete Recording_1 reporting
(36 segments). Pixel AP averages 35 positive segments; IoU averages all 36. These metrics
use the original binary human reference masks, not teacher confidences. No morphology,
external-dataset evaluation, or new Qwen calls.

Preflight independently rasterized every training and validation confidence mask,
confirmed all default hard and ignored-uncertainty training masks remain unchanged,
and matched the baseline initialization hash. A real partially padded training window
matched the fractional-BCE formula and its (sigmoid(logit)-target)/N gradient, with zero
padding gradients and a frozen, gradient-free backbone.

`manifest.json` pins labels, previous results/figures/checkpoints, and implementation
hashes. `source_before/` preserves prior data/trainer/summary code and unchanged model
code for provenance. The shared label-ablation summarizer checks matching initialization
and sampling/RNG hashes across all five epochs, then reproduces calibration and metrics.
Only three dense prediction caches are retained, alongside exact scores and probability
hashes for all 77 segments. Outputs: `comparison.csv`, `summary.json`, and
`confidence_targets_ap.{png,pdf,svg}`. Historical figures are not overwritten.

Runner: `scripts/run_confidence_ablation.sh`.
Service: `birdsong-confidence-ablation-20260910.service`.
Qwen remains manually paused; this runner never starts it.
