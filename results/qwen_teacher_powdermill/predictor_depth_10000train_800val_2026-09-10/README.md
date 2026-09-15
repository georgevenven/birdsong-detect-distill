# Predictor depth with frozen SongMAE-Large

Train one new predictor with two independently initialized transformer layers, d=384
(3,062,432 trainable parameters). Each layer retains four attention heads, feedforward
width 768, GELU, pre-normalization, and dropout 0.1. Both layers use the same padding mask.
Input projection, positional embeddings, and final per-patch output remain unchanged.
Reuse the completed one-layer d128, d384, and d768 controls; do not overwrite their results.

Same 10,000 s self-reviewed XC training set and recording-disjoint 800 s XC validation
set, seed 0, five epochs, checkpoint selected by minimum XC validation BCE. The reused
d128 checkpoint was saved during 15 epochs but is also the minimum within its first
five epochs (epoch 3, constant learning rate). Hard-mask BCE without soft targets or TV;
AdamW lr 0.001, weight decay 0.0001, batch 16, no scheduler or backbone finetuning.
Additional parameters change RNG consumption and minibatch permutations despite the
same seed and fixed data membership. This is a single-seed exploratory comparison.

Powdermill only: calibrate one threshold per model on Recordings_2–4 (41 segments), then
report complete Recording_1 (36 segments). Probability smoothing stays at 2 mel bins /
15 ms, without morphology. Pixel AP uses continuous scores and averages 35 positive
segments; 2D IoU averages all 36 reporting segments. No external-dataset evaluation.

Depth support defaults to one layer; old checkpoint keys and initialization order are
preserved. Archived pre-change source hashes match the historical experiment. Read-only
real-data checks confirmed bit-identical one-layer initialization, RNG state, and FP32 /
FP16 outputs. A two-layer, 16-window, full-length forward/backward/AdamW step had finite
loss and gradients; all backbone parameters remained frozen and gradient-free.

`manifest.json` pins labels, controls, historical figures, before/after implementation
hashes, and current code. Only model-depth handling, checkpoint metadata/loading, and
experiment routing changed; audio preprocessing, optimization loop, scoring and smoothing
are unchanged. `source_before/` archives the three extended shared files for provenance.
Only three dense prediction caches are retained, plus exact scores and probability hashes
for all 77 segments. Outputs: `comparison.csv`, `summary.json`, `predictor_depth_ap.{png,pdf,svg}`.

Runner: `scripts/run_predictor_depth.sh`.
Service: `birdsong-predictor-depth-20260910.service`.
Qwen remains manually paused; this runner never restarts it.
