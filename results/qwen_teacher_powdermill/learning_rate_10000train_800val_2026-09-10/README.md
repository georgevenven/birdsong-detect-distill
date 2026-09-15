# Learning-rate ablation

Compare AdamW 0.001 versus 0.0003, reusing the completed hard-mask Large+d384 control.
Only learning rate changes: frozen SongMAE-Large, one d384 transformer layer, dropout 0.1,
weight decay 0.0001, batch 16, seed 0, and best checkpoint within five epochs by XC validation BCE.
Exactly 10,000 s of self-reviewed XC training labels and 800 s from disjoint XC recordings.
Target, uncertain and chorus boxes are hard positives. No confidence targets, ignored uncertainty,
spatial target softening, total-variation penalty, learning-rate schedule or backbone finetuning.

Powdermill only: unchanged native input, probability smoothing (2 mel bins, 15 ms), and one
threshold calibrated on 41 segments from Recordings 2–4. Report complete Recording_1:
pixel AP over 35 positive segments; mean IoU over all 36 (empty union = 1). Retain three
dense examples and exact scores/probability hashes for all segments. No external evaluation.

The shared trainer now accepts --learning-rate; its default stays 0.001. No other training
operations change. The evaluator and paired-ablation summarizer add an explicit LR condition.
source_before/ archives those three scripts; manifest.json pins inputs, code, prior checkpoints,
reports and figures. Final checks compare initialization and all five epochs' sampling/RNG hashes,
optimizer settings, supervision policies, checkpoint selection, human references and metrics.

Queue behind birdsong-yolo11l-external-20260910.service without changing its files or processes.
Run via birdsong-learning-rate-20260910.service; Qwen remains paused throughout.
Outputs: comparison.csv, summary.json, learning_rate_ap.{png,pdf,svg}.
Single seed and equal epoch budgets: this does not establish which learning rate wins at convergence.
