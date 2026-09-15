# Calibrated teacher–student stage comparison

| VLM annotation pipeline (cumulative) | Teacher pixel AP | Teacher 2D IoU | Student pixel AP | Student 2D IoU |
| --- | ---: | ---: | ---: | ---: |
| Initial prediction (with reasoning) | 0.382 | 0.338 | 0.790 | 0.491 |
| + Self-review | 0.464 | 0.388 | 0.793 | 0.513 |
| + Shifted-context review and reconciliation | 0.460 | 0.383 | 0.794 | 0.514 |
| + Conditional adjudication | 0.462 | 0.382 | 0.792 | 0.516 |

Each student uses frozen SongMAE-Large, identical 10,000 s XC training audio, hard targets and plain BCE.
The checkpoint with minimum BCE on 800 s of recording-disjoint XC validation audio is selected from three epochs.
Student probabilities are Gaussian-smoothed (2 mel bins, 3 frames/15 ms), then thresholded once; no other cleanup.
Each teacher/student threshold maximizes mean 2D IoU on the same 41 segments from Recordings_2–4.
All scores report complete Recording_1: 36 segments, 10,800 s; AP averages its 35 positive segments.
AP uses continuous scores before thresholding. Teacher scores are maximum box confidences, without smoothing.
Shifted review merges agreeing proposals and retains self-review on disagreement; the full pipeline adjudicates disagreements.
The historical direct baseline is excluded because its no-reasoning setting was not reliably enforced.
