# XC annotation continuation to 27,500 seconds

Requested 2026-09-15: pause the paper evaluations, resume Twins, leave V100
annotating, and collect 27,500 total seconds for a future 25,000/2,500 split.

The 12:35 PDT audit found 3,805 valid reasoning + self-review pairs (19,025 s):
Twins 11,370 s and V100 7,655 s. Every pair passed the existing provenance and
annotation validators; all 3,805 source recordings were unique.

The frozen 50k selection, prompts, model settings, and completed files remain
unchanged. Both backends use 16 slots, 16,384-token contexts, reasoning with axes,
and one numbered self-review. The viewing desktop is not an annotation worker.

`birdsong-qwen-budget-27500-20260915.service` monitors accepted pairs and, once
27,500 seconds are present, gracefully stops both annotation clients. In-flight
calls finish and are saved, so the final total may slightly exceed 27,500 s.
It then stops the Twins model server; the V100 model server is left untouched.
`status.json` tracks progress; `completed.json` records the final window list.
The enabled service survives terminal disconnection and user lingering is on.

No new train/validation split or training is launched here. The future split must
remain source-recording-disjoint and preserve evaluation exclusions. Previous
15k experiments keep their original 2,875-second validation set unchanged.

Do not resume the paper suite while this watcher is active. When returning to
evaluation, stop/disable this watcher first, then start the paper suite; that
suite drains Twins before using its GPUs. V100 remains independent.
