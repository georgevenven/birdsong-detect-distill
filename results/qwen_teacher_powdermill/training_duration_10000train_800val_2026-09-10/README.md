# Training duration at fixed 10,000 s of XC labels

Powdermill-only comparison of best-of-3 versus best-of-15 epochs for frozen SongMAE
Micro, Base and Large, with seeds 0/1/2. Reuse the nine completed 3-epoch baselines;
run nine fresh 15-epoch trainings using unchanged `train.py`. Verify matching initial
head weights and first-three-epoch sampling/RNG hashes, with loss differences <=1e-4.
Only the maximum epoch count changes: AdamW lr 0.001 stays constant, batch 16,
hard targets/plain BCE, 128-dimensional head, dropout 0.1 and weight decay 0.0001.

Fixed self-reviewed labels: exactly 10,000 s / 2,105 windows from 249 XC recordings,
plus the same recording-disjoint 800 s / 169 validation windows from 21 recordings.
Select checkpoints by minimum XC validation BCE, not Powdermill performance.
Powdermill Recordings_2–4 calibrate each IoU threshold; complete Recording_1 supplies
the reported AP/IoU. Keep probability smoothing (2 mel bins, 15 ms) and one threshold.
AP uses continuous scores and averages the same 35 positive segments. No morphology.

No WABAD, Hawaii, XC-AJ or NIPS4Bplus evaluation; no label-budget scaling reruns,
learning-rate tuning, larger adapters or backbone finetuning in this first experiment.
Historical figures are untouched. The user subsequently requested that Qwen remain
paused until explicitly restarted. The manual marker at
`/home/george-vengrovski/.config/birdsong-detect-distill/qwen.paused` prevents automatic
server/client restarts, including the existing scheduler's exit trap. Completed stages
are retained; interrupted model calls can be retried on explicit resumption. Keep only three dense
prediction examples per new run; retain exact metrics and probability hashes for all segments.

`manifest.json` pins baselines, data and code. Outputs: `summary.json`, `per_seed.csv`,
`learning_curves.csv`, `caption.txt`, and square `training_duration_ap.{png,pdf,svg}`.
Error bars are sample SD over training seeds with fixed data, not confidence intervals.
