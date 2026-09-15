# Updated Powdermill figures: hard targets and plain BCE

All student curves/bars use hard foreground masks, plain BCE, frozen SongMAE backbones, seed 0, AdamW (lr 0.001, weight decay 0.0001), batch size 16, and the final checkpoint after three epochs. The existing self-review subsets are unchanged: 100/1,000/10,000 seconds. Micro and Base are retrained at 10,000 seconds; Large at 10,000 seconds reuses the completed hard-BCE ablation checkpoint.

| Setting | Pixel AP |
|---|---:|
| Large, 100 s | 0.538084 |
| Large, 1,000 s | 0.705399 |
| Large, 10,000 s | 0.797377 |
| Micro, 10,000 s | 0.733877 |
| Base, 10,000 s | 0.752200 |

The new 90 prediction caches passed checksums, and all 120 new condition/segment masks reproduced saved TP/FP/FN by direct threshold comparisons. Both panels share the exact same Large 10k checkpoint/result. The combined figure and all five qualitative examples were visually inspected and copied, checksum-matched, into `~/Downloads/birdsong_paper_update_2026-09-08/` on gardner-lambda. Display PNG/PDF/SVG files are available there; full probability caches remain on the training machine.

The combined figure shows only pixel AP and starts at 100 seconds. All AP values use continuous probability-space Gaussian-smoothed scores, sigma=(2 mel bins, 3 frames), on the same 30 Recording_1 segments / 8,235 seconds. AP does not depend on a binary operating threshold. The historical Qwen_teacher labels reference remains on the same coverage; additional annotation progress does not change these intervals or teacher values.

Qualitative examples retain the same five excerpts A–E. Ground truth is above the new Large 10k predictions. Contours come directly from the evaluated masks: smooth the full recording, threshold at the frozen 0.08, then crop; no morphology or contour cleanup. Saved mask counts are checked against the ablation evaluation. PNG/PDF/SVG are square, 3.5 inches at 600 dpi, in the standard Matplotlib style.

Historical figures and checkpoints are preserved. The older teacher-stage student table in the manuscript still describes soft-target/TV training with raw inference; do not substitute only its self-review row with the new recipe. A consistently refreshed stage table would require its other student models to be retrained and reevaluated too.

The manuscript should now distinguish 10,000 seconds from 10,000 five-second windows (50,000 seconds). This 10k-second setup has 2,103 windows from 266 XC recordings and uses all windows for three epochs, not a 25% validation split. The Large adapter has 365,088 trainable parameters (approximately 0.37M). Hawaii supplies time–frequency labels and belongs in the 2D evaluation section. The local XC-AJ SVL annotations also contain frequency bounds; using it in the temporal table is a protocol choice, not evidence that its labels are temporal-only. NIPS4Bplus is temporal-only. The external tables now use pixel/frame area metrics, not AP@0.5 or event F1.
