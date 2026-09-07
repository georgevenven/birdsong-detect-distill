# Qwen pipeline ablation

Ten XCL recordings contributed 26 five-second windows, with 1–4 windows per recording. Every condition used the same recordings,
tiles, SongMAE Large 32×1 backbone, head initialization, training order, and three training epochs. Powdermill was the only scoring
corpus. Scores are recording means over all 77 recordings with no post-processing; the development threshold maximizing mean 2D
IoU was 0.02 for every condition.

| Condition | Qwen calls | Temporal IoU | 2D IoU | Foreground boxes |
|---|---:|---:|---:|---:|
| Direct, no reasoning | 1 | 0.6557 | 0.1966 | 61 |
| + private reasoning | 1 | 0.6557 | 0.1961 | 44 |
| + self-review | 2 | 0.6557 | **0.1967** | 49 |
| + shifted review | 3 | 0.6557 | 0.1967 | 49 |
| + conditional adjudication | 3–4 | 0.6557 | 0.1966 | 50 |

The annotations changed materially: compared with direct annotation, 22 of 26 reasoning windows and 23 of 26 windows in each
reviewed condition had different event lists. Shifted review agreed on 14 windows; 12 required adjudication. Despite those label
changes, downstream temporal IoU was identical and the complete 2D IoU range was only 0.0006. This run provides no downstream
evidence that private reasoning, self-review, shifted review, or adjudication improves the detector at this data scale. Direct
annotation is therefore the efficiency baseline; the tiny self-review advantage is too small to interpret as a real gain.

The overlap audit found no exact recording-ID overlap between the existing 411-recording Qwen subset and XCAJ, Powdermill, or
WABAD. The full XCL training pool is not safe without filtering: it contains 406 of 648 XCAJ training IDs and 193 of 319 XCAJ
validation IDs. The ablation manifest excludes all 967 XCAJ IDs.
