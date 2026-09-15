# Gallery update: fine-tuned SongMAE-Large, 3,205 s

All 50 existing examples were regenerated with the seed-0 fully fine-tuned Large encoder and bare linear head. Checkpoint epoch 3 was selected using the unchanged 800-second XC validation set. Source checkpoint and implementation hashes are pinned in `manifest.json`.

Same ten excerpts per dataset: Powdermill, WABAD, Hawaii, XC-AJ, NIPS4Bplus. Original reference images, source audio, crop times, spectrograms and display scales are unchanged. Earlier prediction images remain preserved. No examples were selected using the new model's predictions, and this is not a new dataset-level benchmark.

Inference restores the full trained detection encoder, including CNN and positional embeddings. It uses full-source overlapping-window inference, Gaussian probability smoothing with sigma (2 mel bins, 3 frames), and threshold 0.06 selected on Powdermill Recordings 2–4. Powdermill gallery examples come from Recording 1. No extra mask cleanup or threshold tuning on the examples.

`arrays/` retains each excerpt's smoothed probabilities and exact thresholded mask. Every image pair and unchanged-reference check passed; `complete.json` records completion. The website metadata identifies the new model, budget, selected epoch, and threshold.

Generator: `scripts/update_annotation_gallery_finetuned.py` in the parent repository. Publication details are recorded separately in `deployment.json` after terminal deployment verification.
