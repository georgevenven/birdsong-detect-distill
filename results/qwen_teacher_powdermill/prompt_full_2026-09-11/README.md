# Full Powdermill extension

All five conditions from `../prompt_250_2026-09-11`, extended without prompt,
image, output-schema, or inference changes to all 77 five-minute segments:
4,620 consecutive five-second windows / 23,100 seconds. The extra centered-STFT
endpoint frame is not extra audio and is not annotated as a separate window.

Execution order: reasoning with axes, first numbered review, second numbered
review, direct without axes, direct with axes. All stages are saved separately.
Each review receives only the clean image, a numbered overlay, and the previous
event list. The server uses 16 streams, 16,384 tokens per stream, xhigh reasoning
with a 4,096-token budget and 8,192-token output limit. No-reasoning calls use
explicit `none` / disabled thinking. The pilot selected axes; that decision is
fixed, not repeated using full-data results.

The 250 pilot outputs per condition are reused with source hashes. The original
files remain unchanged. Other windows receive deterministic unique seeds.
Images are rendered on demand from frozen source arrays and discarded after
each call; their hashes are retained. This avoids duplicating ~7.6 GB of PNGs.

Score each complete stage automatically: calibrate its IoU threshold on full
Recordings 2–4; report full Recording 1. Also retain calibration metrics. No
window is dropped after inference failure. The job makes up to three technical
attempts, then saves the error and drains in-flight calls before shutting down.

Run with the user service `birdsong-qwen-full-powdermill-20260911.service`.
It survives terminal disconnection, not shutdown/reboot. A graceful stop drains
in-flight calls; manual restart validates inputs and reuses completed outputs.
No automatic XC resumption. Extra remote GPUs are not yet configured.
