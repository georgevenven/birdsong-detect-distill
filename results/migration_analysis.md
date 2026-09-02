# Migration reproduction

The portable detector loads frozen `georgeven/songmae-large-32x1` weights from Hugging Face. A 365,088-parameter head applies one two-dimensional self-attention layer to the 4×1000 patch grid and emits a 128×1000 mask. Training uses unweighted BCE, light target smoothing, and total-variation regularization.

The recording-disjoint split contained 1,655 training windows and 691 validation windows. Epoch 1 had the lowest validation loss. The untouched half of validation recordings scored 0.545 micro F1, 0.471 precision, 0.645 recall, and 0.495 recording-macro F1 against the Qwen teacher.

## Detection-only transfer

| Dataset | Model | Frame AP | Event AP @ .2 | Event AP @ .5 |
|---|---|---:|---:|---:|
| Powdermill | BirdCODE | 0.946 | 0.226 | 0.098 |
| Powdermill | previous SongMAE 32×4 | 0.946 | 0.240 | 0.115 |
| Powdermill | portable SongMAE 32×1 | 0.941 | 0.218 | 0.102 |
| XCSL | BirdCODE | 0.840 | 0.626 | 0.418 |
| XCSL | previous SongMAE 32×4 | 0.889 | 0.736 | 0.577 |
| XCSL | portable SongMAE 32×1 | 0.892 | 0.730 | 0.578 |
| WABAD site macro | BirdCODE | 0.830 | 0.284 | 0.122 |
| WABAD site macro | previous SongMAE 32×4 | 0.832 | 0.257 | 0.120 |
| WABAD site macro | portable SongMAE 32×1 | 0.826 | 0.243 | 0.110 |

The Large 32×1 reproduction is within 0.005 frame AP and 0.014 strict event AP of the previous local 32×4 detector on Powdermill. On XCSL, strict event AP differs by less than 0.001. WABAD differs by 0.006 frame AP and 0.010 strict event AP. This is a close, but not numerically identical, reproduction because both the Hugging Face backbone variant and patch geometry changed.
