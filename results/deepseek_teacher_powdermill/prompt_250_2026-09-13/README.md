# DeepSeek Flash Vision Exp: matched Powdermill pilot

The user authorized a 250-window, five-condition pilot on September 13, 2026.
The user subsequently approved Zen's available Flash Vision Exp route without
claiming V4.1 identity. The detached pilot service was launched after capability checks.
The screenshot confirms $6.74 in Zen balance and auto-reload disabled. The intended
experiment cap is $5.50 including probes, retries, and outstanding requests.

## Matched experimental design

- Reuse `results/qwen_teacher_powdermill/prompt_250_2026-09-11/manifest.json`;
  SHA256 `3d734d3fa9468fe335493c850eb2e6601c9b344574bb1bdd1cd0549f879f936f`.
- All 250 original windows: 150 Recording_1 reporting, 100 Recordings_2–4 calibration.
- Conditions: direct/plain, direct/axes, reasoning/axes, reasoning plus one numbered
  self-review, reasoning plus two numbered self-reviews. Reviews use this teacher's
  own prior result, not Qwen's detections. Reuse initial outputs between reviews.
- Same source PNGs, system prompt, amended numbered-review prompt, single foreground
  label, confidence, strict local event validation, rasterization, and scorer.
- Keep Qwen's chosen axes for the reasoning chain; do not reselect using report data.
- Calibrate each condition's IoU threshold on the same 100 calibration windows.
  Pixel AP uses continuous box-confidence masks before thresholding, without smoothing.
- DeepSeek controls differ: high reasoning versus disabled; JSON-object mode instead
  of constrained JSON schema; no supported Qwen-style reasoning-token budget or seed.
  Explicitly document provider processing and these differences, not identical settings.
- Initially use 4 concurrent requests; increase to 16 then 32 only after valid output,
  reasoning-mode checks, billing checks, and no throttling. Respect Retry-After.
- Persist charges/reservations before submission, include in-flight worst-case costs,
  and reserve uncertain transport failures as potentially charged. No automatic top-ups.

## Model identity and probes

The live Zen model catalogue lists `deepseek-v4-flash-vision-exp`, but not
`deepseek-v4.1-flash`. The latter is listed on Go, whose subscription documentation
expects coding-agent traffic. No Go requests or subscriptions were made.

A paid Zen request for `deepseek-v4.1-flash` returned HTTP 401 with ModelError:
`Model deepseek-v4.1-flash is not supported` (not an invalid-key error).

A direct, non-reasoning request to Zen's `deepseek-v4-flash-vision-exp` succeeded:
3.29 seconds, 843 prompt tokens, 391 output tokens, 11 locally valid events,
no reported reasoning. Its returned model name was the same legacy alias. This
does not independently establish that Zen served V4.1 weights. DeepSeek's own API
documents legacy alias migration, but that alone does not verify a gateway route.

The first reasoning probe on the same window finished in 51.61 seconds but exhausted
all 8,192 output tokens in reasoning and returned no final JSON (`finish_reason=length`).
It exposed no upstream-model headers. A later probe allowed 32,768 output tokens
and returned valid annotations in 36.78 seconds, using 4,266 output tokens including
4,119 reasoning tokens. The production request caps are 32,768 (retry 49,152) for
reasoning/reviews and 4,096 (retry 6,144) for non-reasoning. At most three attempts
are allowed per condition/window, including format or transport failures.
The three successful HTTP responses used 2,689 input and 12,849 output tokens:
about $0.0081 at DeepSeek's published off-peak uncached rates, not an invoice.

`capability_probes/` preserves individual responses and usage. Successful probes
with identical request hashes are reused, with source hashes retained. Failed probes
remain in the cost ledger but are not detections.

`scripts/probe_deepseek_powdermill.py` makes at most four probes per output folder.
It reads the credential from a private mode-0600 file in the user's runtime directory,
outside the repository; the credential is not embedded in code or result files.
Local Qwen jobs and all frozen inputs remain untouched.

## Detached execution

Service: `birdsong-deepseek-powdermill-pilot-20260913.service` (user systemd).
Runner: `scripts/run_deepseek_powdermill_pilot.py`. No GPU is visible to this process.
It uses four calibration windows to check every condition, then ramps through 16
to 32 concurrent requests. A shared cooldown honors provider throttling. Reviews
use only this teacher's saved final annotations and regenerated numbered overlays.

`budget.sqlite3` records durable worst-case reservations **before** paid requests.
Accounted costs use .15/.60 per million off-peak or .30/1.20 peak, conservatively
above Zen's listed .14/.28 rates, without cache discounts. In-flight requests reserve
peak rates and an input upper bound. Missing usage retains its whole reservation.
All probes and failed attempts count against the $5.50 cap. This is a conservative
token-price estimate, not direct access to account invoices; no billing settings change.

`runtime_amendment.json` documents one parser-only compatibility change: ignore
exactly a root-level `"type": "object"` alongside `events`; all detections retain
strict validation. Full responses remain in attempts. Original runner and manifest
provenance are retained. The warmup was gracefully drained before restarting with
this change; no completed annotations were discarded or overwritten.

`status.json` reports coverage and accounting at completion boundaries;
`budget.sqlite3` is authoritative for currently outstanding reservations.
Each completed condition is automatically scored and added to `comparison.md`
alongside the matched Qwen pilot, not the full-Powdermill scores. A failure or budget
ceiling drains in-flight work and pauses; it does not silently score a subset.
The private temporary credential is removed automatically on full completion.

References: [Zen API and billing](https://opencode.ai/docs/zen/),
[Go scope](https://opencode.ai/docs/go/#where-can-i-use-it),
[DeepSeek model aliases and pricing](https://api-docs.deepseek.com/quick_start/pricing/).
