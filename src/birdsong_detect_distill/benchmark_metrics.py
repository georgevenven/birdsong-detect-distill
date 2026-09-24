"""Class-agnostic area metrics and optional native-event scoring on explicit intervals."""
import librosa
import numpy as np
from scipy import ndimage

from .evaluation import average_precision, matches

RATE = 200
THRESHOLDS = np.linspace(0, 1, 101)
MEL_EDGES = np.linspace(librosa.hz_to_mel(20), librosa.hz_to_mel(16000), 129)
BANDS = {"full": (20, 16000), "common": (500, 12000)}


def rasterize(boxes, width):
    result = np.zeros((128, width), np.float32)
    for left, right, low, high, confidence in np.asarray(boxes).reshape(-1, 5):
        x0, x1 = max(0, int(np.floor(left * RATE))), min(width, int(np.ceil(right * RATE)))
        f = np.clip([low, high], 20, 16000)
        y = (librosa.hz_to_mel(f) - MEL_EDGES[0]) / (MEL_EDGES[-1] - MEL_EDGES[0]) * 128
        y0, y1 = max(0, int(np.floor(y[0]))), min(128, int(np.ceil(y[1])))
        if x1 > x0 and y1 > y0:
            result[y0:y1, x0:x1] = np.maximum(result[y0:y1, x0:x1], confidence)
    return result


def occupancy(score, rate, width):
    output = np.zeros(width, np.float32)
    for index, value in enumerate(score):
        start, end = int(np.floor(index / rate * RATE)), int(np.ceil((index + 1) / rate * RATE))
        output[start:min(end, width)] = np.maximum(output[start:min(end, width)], value)
    return output


def intervals_mask(intervals, width, ignored=()):
    kept = np.zeros(width, bool)
    for start, end in intervals:
        kept[max(0, round(start * RATE)):min(width, round(end * RATE))] = True
    for start, end in ignored:
        kept[max(0, int(np.floor(start * RATE))):min(width, int(np.ceil(end * RATE)))] = False
    return kept


def area(probability, truth):
    p, y = np.asarray(probability).ravel(), np.asarray(truth, bool).ravel()
    if not len(p) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("invalid probability map or empty scoring coverage")
    levels, inverse = np.unique(p, return_inverse=True)
    positive = np.bincount(inverse[y], minlength=len(levels))
    total = np.bincount(inverse, minlength=len(levels))
    tp, predicted = np.cumsum(positive[::-1]), np.cumsum(total[::-1])
    ap = float(np.sum(positive[::-1] * tp / predicted) / positive.sum()) if positive.sum() else None
    bins = np.searchsorted(THRESHOLDS, levels, side="right") - 1
    positive_grid = np.bincount(bins, weights=positive, minlength=len(THRESHOLDS))
    total_grid = np.bincount(bins, weights=total, minlength=len(THRESHOLDS))
    tp = np.cumsum(positive_grid[::-1])[::-1]
    predicted = np.cumsum(total_grid[::-1])[::-1]
    counts = np.stack((tp, predicted - tp, y.sum() - tp), axis=1)
    union = counts.sum(axis=1)
    iou = np.divide(tp, union, out=np.ones(len(tp)), where=union != 0)
    return dict(ap=ap, iou_curve=iou.tolist(), counts=counts.astype(np.int64).tolist())


def area_scores(probability, score, events, width, intervals, ignored=(), temporal_only=False):
    truth = rasterize(np.column_stack((events, np.ones(len(events)))), width).astype(bool)
    kept = intervals_mask(intervals, width, ignored)
    output = {"temporal": area(score[kept], truth.any(axis=0)[kept])}
    if probability is not None and not temporal_only:
        hz = librosa.mel_to_hz(MEL_EDGES)
        for name, (low, high) in BANDS.items():
            rows = (hz[:-1] < high) & (hz[1:] > low)
            output[name] = area(probability[rows][:, kept], truth[rows][:, kept])
    return output


def crop_events(events, intervals):
    return np.asarray([[max(start, left), min(end, right)] for start, end in np.asarray(events).reshape(-1, 2)
        for left, right in intervals if min(end, right) > max(start, left)]).reshape(-1, 2)


def nms(boxes, threshold=.7):
    """Cross-window YOLO deduplication in time-frequency, not across temporal streams."""
    boxes = np.asarray(boxes).reshape(-1, 5)
    order, kept = np.argsort(-boxes[:, 4], kind="stable"), []
    size = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    while len(order):
        index, order = order[0], order[1:]
        kept.append(index)
        overlap = np.maximum(0, np.minimum(boxes[index, 1], boxes[order, 1]) - np.maximum(boxes[index, 0], boxes[order, 0]))
        overlap *= np.maximum(0, np.minimum(boxes[index, 3], boxes[order, 3]) - np.maximum(boxes[index, 2], boxes[order, 2]))
        iou = overlap / np.maximum(size[index] + size[order] - overlap, 1e-12)
        order = order[iou <= threshold]
    return boxes[kept]


def component_events(probability, threshold):
    labels, _ = ndimage.label(probability >= threshold, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    return np.asarray([[x[1].start / RATE, x[1].stop / RATE] for i, x in enumerate(ndimage.find_objects(labels), 1)
        if x is not None and sizes[i] >= 32 and x[1].stop - x[1].start >= 3]).reshape(-1, 2)


def event_counts(estimates, references, intervals):
    truth = crop_events(references, intervals)
    output = {str(iou): [] for iou in (.2, .5)}
    for events in estimates:
        estimate = crop_events(events, intervals)
        for iou in (.2, .5):
            output[str(iou)].append(matches(truth, estimate, iou).tolist())
    if any(len(counts) != len(THRESHOLDS) for counts in output.values()):
        raise ValueError("event threshold sweep is incomplete")
    return output


def summarize(rows, thresholds):
    output = {}
    for metric in rows[0]["area"]:
        values = [row["area"][metric] for row in rows]
        ap = [v["ap"] for v in values if v["ap"] is not None]
        index = thresholds[metric]
        tp, fp, fn = np.sum([v["counts"][index] for v in values], axis=0)
        output[metric] = dict(ap=float(np.mean(ap)) if ap else None, positive_segments=len(ap),
            iou=float(np.mean([v["iou_curve"][index] for v in values])), threshold=float(THRESHOLDS[index]),
            pooled_precision=float(tp / max(1, tp + fp)), pooled_recall=float(tp / max(1, tp + fn)))
    if "events" in rows[0]:
        output["events"] = {}
        for iou in ("0.2", "0.5"):
            counts = np.sum([row["events"][iou] for row in rows], axis=0)
            index = thresholds[f"event_{iou}"]
            tp, fp, fn = counts[index]
            output["events"][iou] = dict(ap=average_precision(counts) if counts[0, 0] + counts[0, 2] else None,
                f1=float(2 * tp / max(1, 2 * tp + fp + fn)),
                threshold=float(THRESHOLDS[index]), counts=counts[index].tolist())
    return output


def calibrate(rows):
    thresholds = {key: int(np.argmax(np.mean([r["area"][key]["iou_curve"] for r in rows], axis=0)))
        for key in rows[0]["area"]}
    if "events" in rows[0]:
        for iou in ("0.2", "0.5"):
            counts = np.sum([r["events"][iou] for r in rows], axis=0)
            tp, fp, fn = counts.T
            thresholds[f"event_{iou}"] = int(np.argmax(2 * tp / np.maximum(2 * tp + fp + fn, 1)))
    return thresholds
