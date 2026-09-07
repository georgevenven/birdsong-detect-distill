# SongMAE 32×1 Qwen scaling on Powdermill

## Default metrics

The main results use only two metrics:

- **Temporal IoU:** collapse frequency with a maximum, then compare bird-present time against the union of annotated bird-present time.
- **2D IoU:** compare the predicted and annotated unions directly on the 200 Hz × 128 mel-bin time–frequency grid.

Both are means over the 77 Powdermill recordings. For each model, one probability threshold is selected on Powdermill development data by maximizing recording-mean 2D IoU; that same threshold is used for temporal IoU. There is no smoothing, morphology, connected-component conversion, AP, precision/recall, or instance-box matching in the default evaluation. An empty prediction and empty reference have IoU 1.

The Qwen subsets are deterministic, nested recording sets at seed 0. Every accepted window from each selected recording was used for three epochs with a frozen SongMAE backbone.

| Qwen recordings | Accepted windows | Qwen boxes |
|---:|---:|---:|
| 10 | 148 | 714 |
| 50 | 421 | 2,045 |
| 100 | 751 | 3,948 |
| 300 | 2,708 | 14,342 |

## Results

Each cell is `temporal IoU / 2D IoU`.

| Qwen recordings | Micro, 1.75M | Base, 14.89M | Large, 98.65M |
|---:|---:|---:|---:|
| 10 | .656 / .281 | .697 / .431 | .722 / .427 |
| 50 | .664 / .403 | .761 / **.488** | .744 / .470 |
| 100 | .662 / .399 | .727 / .483 | .740 / .481 |
| 300 | .721 / .431 | **.764** / .485 | .758 / .488 |

Base trained on 300 recordings has the best temporal IoU (.764). Base/50 has the best 2D IoU (.4879), effectively tied with Large/300 (.4876). Micro/300 reaches .721 temporal IoU and .431 2D IoU while using roughly 56× fewer backbone parameters than Large.

The plotting style and exact backbone parameter counts follow the latest remote revision of SongMAE's `plot_beans_parameters.py` (`origin/main`, commit `1cd3590`).
