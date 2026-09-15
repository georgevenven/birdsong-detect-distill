"""Native inference adapters. Frequency masks and species streams stay intact until scoring."""
import importlib.metadata
import json
from itertools import batched
from pathlib import Path

import librosa
import numpy as np
import torch

from .benchmark_data import digest
from .benchmark_metrics import MEL_EDGES, RATE, THRESHOLDS, component_events, nms, occupancy, rasterize
from .birdbox import REVISION, model_path, predict_boxes

BIRDCODE_REPO = "EarthSpeciesProject/sed-birdcode"
BIRDCODE_REVISION = "b7a8ee261579278f44f636baf4024ce384ceb490"
BIRDCODE_CODE = "617c6408fdd1b56c3873ef2d17006ffbaee61783"


class Predictor:
    def __init__(self, kind, checkpoint, device, batch_size=8, confidence=.0001, max_det=3000, variant="yolo11n", backbone_revision=None):
        self.kind, self.device, self.batch_size = kind, device, batch_size
        self.confidence, self.max_det = confidence, max_det
        self.metadata = dict(model=kind, batch_size=batch_size, precision="float32_native_except_SongMAE_float16_autocast",
            postprocessing="native_model_output_no_SongMAE_cleanup", channel="first" if kind == "birdbox" else "mean",
            resampling="native_rate" if kind == "birdbox" else "kaiser_best_scale_true" if kind == "birdcode" else "soxr_hq_scale_true")
        if kind == "birdcode":
            from huggingface_hub import snapshot_download
            from sound_event_detection.models import FrameDetector
            provenance = json.loads(importlib.metadata.distribution("sound-event-detection").read_text("direct_url.json") or "{}")
            if provenance.get("vcs_info", {}).get("commit_id") != BIRDCODE_CODE:
                raise ValueError("install the pinned BirdCODE code from requirements/birdcode.txt")
            folder = Path(snapshot_download(BIRDCODE_REPO, revision=BIRDCODE_REVISION))
            checkpoint = folder / "checkpoint/best_model.pt"
            self.model = FrameDetector.from_hf_hub(BIRDCODE_REPO, revision=BIRDCODE_REVISION).eval().to(device)
            self.metadata.update(repo=BIRDCODE_REPO, revision=BIRDCODE_REVISION, upstream_code=BIRDCODE_CODE,
                frame_rate=self.model.frame_rate, classes=len(self.model.labels), window_seconds=5, overlap=.5,
                event_postprocessing=dict(merge_max_gap=1.0, min_event_duration=.01, nms_iou=.8),
                tail_policy="native zero-padded windows; retain partial final frame before cropping to original duration",
                species_filter="none: all bird outputs; no ground-truth species oracle")
        elif kind in {"birdbox", "qwen_yolo"}:
            from ultralytics import YOLO
            checkpoint = model_path(variant, Path("models")) if kind == "birdbox" else checkpoint
            if checkpoint is None:
                raise ValueError("teacher-trained YOLO needs --checkpoint")
            self.model = YOLO(checkpoint)
            self.metadata.update(confidence_floor=confidence, nms_iou=.7, max_det=max_det,
                upstream_code=REVISION if kind == "birdbox" else None,
                input_representation="linear_Hz_magma_native_BirdBox" if kind == "birdbox" else "128_mel_viridis_5s",
                window_seconds=6 if kind == "birdbox" else 5, stride_seconds=5 if kind == "birdbox" else 2.5)
            if kind == "qwen_yolo":
                sidecar = Path(checkpoint).with_suffix(".json")
                self.metadata["training"] = json.loads(sidecar.read_text()) if sidecar.exists() else None
        elif kind == "songmae":
            from .model import load_detector
            if checkpoint is None:
                raise ValueError("SongMAE needs --checkpoint")
            if not backbone_revision:
                raise ValueError("pin --backbone-revision for a reproducible SongMAE comparison")
            self.backbone, self.head, saved = load_detector(checkpoint, torch.device(device), revision=backbone_revision)
            self.metadata.update(backbone=saved["backbone_id"], backbone_revision=self.backbone.config._commit_hash,
                training=saved["metrics"], window_seconds=5, stride_seconds=2.5, overlap_aggregation="maximum",
                event_postprocessing=dict(connectivity=8, minimum_area=32, minimum_frames=3, smoothing="none"))
        else:
            raise ValueError(kind)
        self.metadata.update(checkpoint_sha256=digest(checkpoint), checkpoint=str(checkpoint))
        self.metadata["versions"] = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "librosa", "torch", "transformers")}
        if kind in {"birdbox", "qwen_yolo"}:
            self.metadata["versions"]["ultralytics"] = importlib.metadata.version("ultralytics")

    @torch.inference_mode()
    def predict(self, audio, sample_rate):
        if self.kind == "birdbox":
            return dict(boxes=predict_boxes(self.model, audio, sample_rate, self.device,
                self.confidence, self.batch_size, self.max_det))
        if self.kind == "birdcode":
            # run() otherwise rounds away the last partially valid coarse frame.
            window, hop = round(5 * sample_rate), round(2.5 * sample_rate)
            length = window + max(0, int(np.ceil((len(audio) - window) / hop))) * hop
            output = self.model.run(np.pad(audio, (0, length - len(audio)))[None],
                batch_size=self.batch_size, device=self.device, overlap=.5)
            frames = int(np.ceil(len(audio) / sample_rate * output.frame_rate))
            return dict(frames=output.predictions[0, :frames], frame_rate=np.asarray(output.frame_rate), labels=np.asarray(output.class_names))
        if self.kind == "songmae":
            from evaluate_2d import songmae_probability
            return dict(probability=songmae_probability(self.backbone, self.head, audio, torch.device(self.device)))
        from evaluate_qwen_yolo import windows
        _, tiles = windows(audio)
        boxes = []
        for batch in batched(tiles, self.batch_size):
            results = self.model.predict([x[2] for x in batch], imgsz=1024, device=self.device,
                conf=self.confidence, iou=.7, max_det=self.max_det, verbose=False)
            for result, (start, valid, _) in zip(results, batch):
                if len(result.boxes) >= self.max_det:
                    raise ValueError("YOLO hit max_det; increase it before reporting AP")
                for (left, top, right, bottom), confidence in zip(result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    low, high = librosa.mel_to_hz(MEL_EDGES[0] + (1 - np.asarray([bottom, top])) * (MEL_EDGES[-1] - MEL_EDGES[0]))
                    boxes.append([(start + left * 1000) / RATE, (start + min(valid, right * 1000)) / RATE, low, high, confidence])
        return dict(boxes=np.asarray([b for b in boxes if b[1] > b[0]], np.float32).reshape(-1, 5))


def maps(prediction, duration):
    width = int(np.ceil(duration * RATE))
    if "frames" in prediction:
        return None, occupancy(prediction["frames"].max(axis=1), float(prediction["frame_rate"]), width)
    probability = rasterize(prediction["boxes"], width) if "boxes" in prediction else prediction["probability"][:, :width]
    if probability.shape != (128, width):
        raise ValueError("prediction length does not cover the recording")
    return probability, probability.max(axis=0)


def native_events(prediction, probability):
    if "frames" in prediction:
        from sound_event_detection.utils.postprocessing import postprocess_selection_table
        from sound_event_detection.utils.reformatters import frames_to_selection_table
        values, rate = prediction["frames"], float(prediction["frame_rate"])
        labels = prediction["labels"].tolist()
        for threshold in THRESHOLDS:
            # Keep separate species streams until AFTER native event processing.
            active = np.flatnonzero(values.max(axis=0) >= threshold)
            probs = values[:, active]
            table = frames_to_selection_table(probs >= threshold, [labels[i] for i in active], rate, probs=probs)
            table = postprocess_selection_table(table, {"merge_max_gap": 1.0, "min_event_duration": .01, "nms": {"iou_threshold": .8}})
            yield table[["Begin Time (s)", "End Time (s)"]].to_numpy(float)
    elif "boxes" in prediction:
        boxes = nms(prediction["boxes"])
        for threshold in THRESHOLDS:
            yield boxes[boxes[:, 4] >= threshold, :2]
    else:
        for threshold in THRESHOLDS:
            yield component_events(probability, threshold)
