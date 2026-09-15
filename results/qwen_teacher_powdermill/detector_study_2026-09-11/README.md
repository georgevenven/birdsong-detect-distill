# Ordered SongMAE detector study

Frozen protocol: `manifest.json`. Progress: `status.json` and `logs/`.

1. **Width:** 128, 384 and 768, three seeds each, original 10,000 s training set. Reuse five verified checkpoints and train four missing seeds. Select the highest mean Powdermill pixel AP; exact ties favor the narrower head.
2. **Label budget:** selected width, fresh nested 10,000/20,000 s self-review subsets, three seeds each. Both subsets have identical proportions of the two teacher annotation batches. Do not substitute the older 10k training set for this comparison.
3. **Pointwise head:** selected width, original 10,000 s, three seeds. Compare zero transformer layers with the one-layer width-study controls. Shared projection/output weights, positional embeddings and CPU RNG initialization are matched; backbone tokens remain contextual.

All runs use frozen SongMAE-Large, hard foreground BCE, no target softening or TV, AdamW at 1e-3, batch 16, and minimum XC validation BCE within five epochs. The fixed 800 s validation set comprises 169 windows from 21 source recordings excluded from every training set.

| Training set | Seconds | Windows | Recordings |
| --- | ---: | ---: | ---: |
| Original 10k | 10,000 | 2,105 | 249 |
| Expanded 10k | 10,000 | 2,096 | 700 |
| Expanded 20k | 20,000 | 4,179 | 1,107 |

Powdermill only: score full Recording_1; choose each mask threshold on Recordings_2–4. Use probability smoothing (2 mel bins, 3 frames), one threshold and no morphology. Pixel AP uses continuous smoothed scores. Report three-seed mean and sample SD; this is not a dataset confidence interval. Positive-only IoU is retained as a diagnostic for the empty-segment convention.

The budget comparison fixes epochs, not optimizer updates: 20k entails approximately twice the training updates. Expanded labels use accepted original-pipeline windows, extracting self-review before shifted review/adjudication; 13 unresolved windows are excluded. These are exploratory development comparisons, not held-out external results.

Outputs: each stage gets `summary.json`, `comparison.csv` and square `pixel_ap.png/pdf/svg` plots. Thirteen new training/evaluation jobs run across two GPU lanes, with a barrier between stages. Historical figures, tables and reused checkpoints are hash-protected. Only three dense prediction arrays per new model are retained to conserve disk space; all segment metrics and prediction hashes are retained.

The independent user service is `birdsong-detector-study-20260911.service`. It survives closing the terminal/chat while this computer stays on. It does not restart Qwen or automatically restart after failure/reboot. Completed conditions can be resumed with the same frozen files; inspect any failed run before restarting.

Preflight passed: all five reused checkpoints and calibrated scores reproduce; old/new one-layer heads produce bit-identical outputs on real XC backbone tokens at all three widths; zero-layer heads preserve shared initialization/RNG and have finite hard-BCE gradients. Syntax compilation passed. No test files were added.
