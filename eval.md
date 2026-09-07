# NIPS4Bplus evaluation protocol

## Role

NIPS4Bplus is evaluation-only. Do not use its audio, weak labels, strong annotations, or scores for training, prompt design, model selection, threshold selection, calibration, early stopping, or post-processing choices. Freeze the model and the complete inference pipeline before evaluating it. Preserve every run; do not select the best NIPS4Bplus result.

## Data

The local copy is under `data/nips4bplus/`:

- `audio/`: all 687 NIPS4B training recordings, mono at 44.1 kHz.
- `annotations/`: 674 strong temporal annotation CSVs plus the class and weak-label tables.
- `source/`: the original audio archive and NIPS4Bplus v7 files.

Score only the 674 recordings with strong annotations. The release omits 13 annotations: six recordings were ambiguous and seven contained only insects. Do not treat those recordings as negative examples.

Map strong labels as follows:

| Reference group | Labels | Events | Recordings | Bird-detection treatment |
| --- | ---: | ---: | ---: | --- |
| Bird | 77 | 5,279 | 539 | Positive |
| Insect | 9 | 168 | 88 | Negative; retain attribution |
| Amphibian | 1 | 34 | 9 | Negative; retain attribution |
| Human | 1 | 12 | 2 | Negative; retain attribution |
| Unknown | 1 | 282 | 99 | Ignore region |

Unannotated time is background. Mask `Unknown` intervals from both positive and negative counts. Keep the original species/call label in all event-level output even when labels are collapsed for scoring.

## Frozen evaluation

Record the model identifier, checkpoint SHA-256, code revision, preprocessing, windowing, score aggregation, and post-processing. A fixed operating threshold must come from a separate development dataset and be declared before this run; never tune it on NIPS4Bplus.

Use the repository's temporal detection protocol for the primary comparison:

- collapse the 77 bird labels into `bird`;
- resample scores to 100 Hz;
- sweep 101 thresholds from 0 to 1 for frame AP and event AP;
- merge predicted spans separated by less than one second and discard spans shorter than 10 ms;
- match events one-to-one and report event AP at temporal IoU 0.2 and 0.5;
- also report precision, recall, F1, and temporal IoU at the predeclared fixed threshold.

Aggregate counts over the scored recordings. Because the dataset is small, report 95% confidence intervals from 10,000 recording-level bootstrap resamples with a fixed seed.

## Non-bird vocalization record

Save every predicted event before reference matching. For each unmatched bird prediction, attach the reference event with greatest temporal intersection, if any. This records whether the model fired on an insect, amphibian, human, unknown, or background sound.

Each prediction record must contain:

`recording`, `model`, `checkpoint_sha256`, `onset_s`, `offset_s`, `score`, `model_output`, `threshold`, `reference_group`, `reference_label`, `reference_onset_s`, `reference_offset_s`, `intersection_s`, `iou`, and `match_status`.

Use `reference_group` values `bird`, `insect`, `amphibian`, `human`, `unknown`, and `background`. Preserve the model's native output semantics: a bird-only prediction overlapping an insect or human is a bird false positive tagged with that reference group, not a correctly classified non-bird vocalization. If a model explicitly emits `other_vocalization`, score that output separately against insect, amphibian, and human events.

Alongside the primary bird metrics, report for each non-bird group:

- annotated events and duration;
- matched predictions and predicted duration at the fixed threshold;
- fraction of reference events and seconds overlapped by any prediction;
- false-positive predictions per annotated minute; and
- score distribution, including median and 95th percentile.

Write summaries and event records under `results/reproduced/nips4bplus/`.

## Provenance

NIPS4Bplus v7 is distributed under CC BY 4.0. Cite Morfi, Stowell, and Pamula, *NIPS4Bplus: Transcriptions of NIPS4B 2013 Bird Challenge Training Dataset*, [Figshare](https://doi.org/10.6084/m9.figshare.6798548), and the associated [PeerJ Computer Science paper](https://doi.org/10.7717/peerj-cs.223). The downloaded Figshare files match the MD5 values published with the release.
