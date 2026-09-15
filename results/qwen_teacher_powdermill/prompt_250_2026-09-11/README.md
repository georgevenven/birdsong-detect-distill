# Simplified Qwen prompt: 250 matched windows

Single-class bird localization, including individual calls/songs and inseparable
choruses. Ambiguous sounds require a visual-evidence judgment rather than an
uncertain class. Confidence now means estimated probability of bird presence,
not certainty in an uncertain/chorus class. Outputs contain only `label`,
`bbox_2d`, and `confidence` inside an `events` list. No summaries, flags,
truncation/overlap indicators, non-bird labels, or individual-ID instructions.
The frozen manifest contains the complete initial and self-review prompts.

## Sequence

1. Fresh no-reasoning initial predictions on plain and numbered-axis images.
2. Select image format by higher calibration pixel AP; exact ties retain plain.
3. Initial predictions with xhigh reasoning, a 4,096-token reasoning budget,
   and an 8,192-token total output limit.
4. One self-review of those reasoning-enabled initial predictions.
5. A second self-review of the once-reviewed predictions.

The last three stages all use xhigh and the same selected image format. Each
review receives the clean image, a red overlay of the immediately previous
event list, and that list as JSON. Red badges numbered 1, 2, 3, … match temporary
`box_id` fields in that input JSON. They are box references, not bird identities.
Returned detections still contain only `label`, `bbox_2d`, and `confidence`;
numbering is regenerated from the updated list for the next review.
Reviews are separate stateless calls, not a
growing conversation. There is no shifted reviewer or adjudicator. All five
conditions have 250 outputs, for 1,250 accepted calls, excluding technical retries.

The user requested numbered reviews before any self-review had run. The original
manifest is preserved; `review_amendment.json` records the added prompt text,
code/font hashes, and unchanged scope of all initial predictions and scoring.
Original scripts are archived in `provenance/before_box_ids`. Review outputs and
scores reference both the original manifest and the amendment; direct outputs
remain reusable without modification. This is not a numbered-versus-unnumbered
review ablation: both review rounds use numbers.

## Images and sampling

Both input conditions use the same 2240×704 canvas, with the same 2048×512
spectrogram pixels at x=128…2176, y=56…568. Only the numbered condition has
ticks and labels outside the plot. There are no interior grid lines. x runs
0–1000 left-to-right, y runs 0–1000 top-to-bottom. JSON order stays
`[x_min,y_min,x_max,y_max]`; values are normalized to the plot, not the PNG.
This controls canvas size, padding, and plot resolution between conditions.
The plain control is not the historical margin-free image format.

Seed 17 selects full five-second windows without looking at references or
predictions. All 77 five-minute segments are represented, with approximately
equal window counts within each partition: 150 windows from Recording 1 for
reporting; 100 from Recordings 2–4 for calibration. The old checkpoint-limited
100-window sample is not used as the baseline.

## Evaluation and interpretation

Reuse the existing 128-mel / 5-ms human-reference rasterization. Teacher scores
are maximum overlapping box confidence, zero elsewhere; no smoothing. Report
segment-macro pixel AP and IoU plus pooled pixel precision and recall on identical
selected intervals. Each condition's threshold maximizes calibration mean IoU
over 0.01–1.00 in steps of 0.01. Zero is excluded prospectively: score >= 0 would
label the whole spectrogram, including pixels where the model returned no box.
Also retain a shared-threshold 0.01 diagnostic. Never choose axes or thresholds
using Recording 1. This is exploratory development evidence, not an independent
held-out test, a full-Powdermill result, or downstream student performance.

Raw new labels stay `bird_vocalization`. Only the internal scoring copy maps
that name to the existing foreground class so production parsers remain unchanged.
The taxonomy, confidence semantics, and prompt differ from historical results;
cross-study improvements cannot isolate the effect of metadata removal alone.

## Execution

`scripts/run_qwen_prompt_study.py --prepare` freezes the inputs; the user service
`birdsong-qwen-prompt-study-20260911.service` runs the complete sequence with a
dedicated 16-slot × 16,384-token server. Template preflights verify both modes;
returned reasoning must be absent for direct calls and present for xhigh calls.
Missing token counters remain unknown; reasoning text itself is not retained.
Each attempt saves latency, usage, final content and mode-verification metadata.
Up to three technical attempts are allowed per output, using the same seed;
failures remain in the audit. No failed windows are silently dropped.

The service survives terminal disconnection while the computer stays on, but
does not restart after reboot. Stopping it stops scheduling and lets in-flight
calls finish before stopping its server. Completed outputs are resumable under
the same manifest. The original XC queue stays paused, and the server stops at
the end. No historical annotations, model weights, paper figures or tables are
overwritten. No more than two self-reviews are scheduled.
