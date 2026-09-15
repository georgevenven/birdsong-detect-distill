# No-reasoning Powdermill pilot

Generate 100 fresh, single-call annotations with explicit `reasoning_effort=none`,
zero reasoning budget, and `enable_thinking=false`. Verify an empty, closed thinking
prefix and audit returned reasoning fields, completion tokens, and latency. Missing
reasoning-token counters are recorded as unknown, not measured zero.

The seed-17, label-blind sample contains 60 evaluation windows from Recording 1
and 40 calibration windows from Recordings 2–4 (14/13/13). Each window is five
seconds. Inputs, original sampling seeds, prompts, historical direct labels, and
scoring code are pinned in `manifest.json`. Historical direct annotations were
not verified non-reasoning: this is a matched historical comparison, not a clean
reasoning-on versus reasoning-off experiment. Only windows with retained stage
checkpoints are eligible; the four known post-fix direct windows are excluded.

Each method is evaluated on the same intervals. Pixel AP uses continuous maximum
foreground-box confidence, without smoothing. Each method's operating threshold
maximizes segment-macro IoU on the same 40 calibration windows. Report AP and IoU
over the 60 evaluation windows, with pooled pixel precision and recall. These
subset results must not replace full-Powdermill paper-table values.

Run `scripts/run_qwen_no_reason_pilot.py` through the dedicated user service
`birdsong-qwen-no-reason-100-20260911.service`. Its temporary server uses 32 slots
of 8,192 context tokens; it is stopped afterward. The original XC annotation
queue and its 16-slot configuration remain paused and unchanged. Outputs are
kept here, separate from all historical labels. The service survives terminal
disconnection while this machine stays on; it does not restart after reboot.

The timing report spans first request to last response, including any retries
and excluding model-server startup. Individual latency differs from aggregate
throughput because requests run concurrently. A 320-second estimate assumes
about 1,600 output tokens per window at a sustained aggregate 500 tokens/s.
