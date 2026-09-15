# Uncertain-vocalization supervision ablation

Compare the completed Large+d384 one-layer seed-0 baseline (uncertain boxes positive)
with one new run that ignores uncertain-only pixels in training and XC validation loss.
Keep target_vocalization and chorus positive, including overlaps with uncertain boxes.
Ignore is encoded as target -1 and masked out before BCE; it is not a background label.
Every original audio window is retained, including windows containing uncertain calls.

Frozen SongMAE-Large, 1,878,560 trainable predictor parameters, 10,000 s XC training
(2,105 windows / 249 recordings) and recording-disjoint 800 s XC validation (169 windows /
21 recordings). Seed 0, five epochs, minimum supervised-pixel XC validation BCE checkpoint.
AdamW lr 0.001, weight decay 0.0001, batch 16, dropout 0.1; hard BCE without smoothing or
TV. Initial weights and each epoch's sampling/RNG hashes must match the existing baseline.
Both policies use their respective validation supervision for checkpoint selection;
raw validation BCE values across policies are therefore not directly comparable.

Audited counts (valid pixels exclude audio padding):

| Partition | Valid pixels | Ignored pixels | Positive pixels retained | Positive overlap retained |
| --- | ---: | ---: | ---: | ---: |
| Training | 256,000,000 | 3,919,609 | 18,671,942 | 5,336 |
| Validation | 20,480,000 | 362,226 | 2,260,159 | 0 |

There are no fully ignored windows. BCE is averaged over supervised pixels; validation
batch means are weighted by their supervised-pixel counts. All-ignored batches are
handled without an optimizer update, but none can occur with this fixed dataset.

Powdermill evaluation remains unchanged: Gaussian probability smoothing (2 mel bins /
15 ms), one threshold maximizing mean 2D IoU on Recordings_2–4 (41 segments), then complete
Recording_1 reporting (36 segments). Pixel AP averages 35 positive segments; IoU averages
all 36. No human reference pixels are ignored. No external-dataset evaluation or Qwen calls.

Preflight checks independently rasterized every training and validation mask, confirmed
unchanged default masks for all training windows, and matched the baseline initialization
hash. A real partially padded training window produced zero gradients at ignored and
padded pixels, finite nonzero supervised gradients, and unchanged default BCE.

`manifest.json` freezes labels, baseline reports/checkpoints, historical figures, and code.
`source_before/` archives the three shared files extended for this opt-in policy. Existing
one-layer controls and all historical results remain untouched. Final outputs are
`comparison.csv`, `summary.json`, and `uncertain_supervision_ap.{png,pdf,svg}`. Keep exact
scores and probability hashes for all 77 segments, with only three dense example caches.

Runner: `scripts/run_uncertain_ablation.sh`.
Service: `birdsong-uncertain-ablation-20260910.service`.
Qwen remains manually paused; this runner never starts it.
