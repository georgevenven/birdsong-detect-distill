"""Shared recording inventory; decoding stays separate from model-specific resampling."""
import csv
import hashlib
import io
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .evaluation import WABAD_SITES


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def members(archive, suffixes):
    return {Path(n).stem: n for n in archive.namelist()
        if Path(n).suffix.lower() in suffixes and not Path(n).name.startswith("._") and "__MACOSX" not in n}


def recordings(root, dataset):
    root = Path(root)
    if dataset == "powdermill":
        archive = root / "powdermill/wav_Files.zip"
        with zipfile.ZipFile(archive) as sounds, zipfile.ZipFile(root / "powdermill/annotation_Files.zip") as labels:
            names = members(sounds, {".wav"})
            for member in sorted(members(labels, {".txt"}).values()):
                name = Path(member).name.split(".Table")[0]
                table = pd.read_csv(labels.open(member), sep="\t")
                events = table[["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]].to_numpy(float)
                yield dict(name=name, archive=str(archive), member=names[name], events=events, group=name.split("_Segment_")[0])
    elif dataset == "xcsl":
        archive = root / "xcaj/Audio.zip"
        with zipfile.ZipFile(archive) as sounds, zipfile.ZipFile(root / "xcaj/Annotation.zip") as labels:
            names = members(sounds, {".wav", ".mp3"})
            for name, member in sorted(members(labels, {".svl"}).items()):
                tree = ET.fromstring(labels.read(member))
                rate = float(tree.find("./data/model").attrib["sampleRate"])
                events = []
                for point in tree.findall("./data/dataset/point"):
                    p = point.attrib
                    events.append([float(p["frame"]) / rate, (float(p["frame"]) + float(p["duration"])) / rate,
                        float(p["value"]), float(p["value"]) + float(p["extent"])])
                yield dict(name=name, archive=str(archive), member=names[name], events=np.asarray(events).reshape(-1, 4), group=name)
    elif dataset == "wabad":
        table = pd.read_csv(root / "wabad/Pooled annotations.csv")
        columns = ["Begin_Time_(s)", "End_Time_(s)", "Low_Freq_(Hz)", "High_Freq_(Hz)"]
        for site in WABAD_SITES:
            archive = root / "wabad" / f"{site}.zip"
            annotations = {name: t[columns].to_numpy(float) for name, t in table[table.Site == site].groupby("Recording")}
            with zipfile.ZipFile(archive) as sounds:
                for _, member in sorted(members(sounds, {".wav"}).items()):
                    name = Path(member).name
                    yield dict(name=f"{site}/{name}", archive=str(archive), member=member,
                        events=annotations.get(name, np.empty((0, 4))), group=site)
    elif dataset == "nips4bplus":
        with (root / "annotations/nips4b_birdchallenge_espece_list.csv").open() as stream:
            types = {r["class name"]: r["type"].lower() for r in csv.DictReader(stream)}
        for path in sorted((root / "audio").glob("*.wav")):
            number = path.stem.removeprefix("nips4b_birds_trainfile")
            label = root / "annotations" / f"annotation_train{number}.csv"
            if not label.exists():
                continue  # Never turn an unannotated recording into a negative example.
            events, ignored = [], []
            with label.open() as stream:
                for row in csv.reader(stream):
                    if len(row) < 3 or not row[2].strip():
                        continue
                    start, end = float(row[0]), float(row[0]) + float(row[1])
                    kind = types.get(row[2].strip(), row[2].strip().lower())
                    if kind == "bird":
                        events.append([start, end, 20, 16000])
                    elif kind == "unknown":
                        ignored.append([start, end])
                    elif kind not in {"insect", "amphibian", "human", "noise"}:
                        raise ValueError(f"unrecognized NIPS4Bplus class: {row[2]}")
            yield dict(name=path.stem, path=str(path), events=np.asarray(events).reshape(-1, 4),
                ignored=ignored, group=path.stem, temporal_only=True)
    else:
        raise ValueError(dataset)


def read_audio(record):
    if "archive" in record:
        with zipfile.ZipFile(record["archive"]) as archive:
            encoded = archive.read(record["member"])
    else:
        encoded = Path(record["path"]).read_bytes()
    try:
        return sf.read(io.BytesIO(encoded), dtype="float64", always_2d=True)
    except sf.LibsndfileError:
        # Decode only: preserve sample rate/channels so the model's resampler remains authoritative.
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "wav", "-c:a", "pcm_f32le", "pipe:1"],
            input=encoded, capture_output=True, check=True).stdout
        return sf.read(io.BytesIO(decoded), dtype="float64", always_2d=True)


def model_audio(waveform, sample_rate, model):
    if model == "birdbox":
        return np.ascontiguousarray(waveform[:, 0]), sample_rate
    value = waveform.astype(np.float32).mean(axis=1)
    if sample_rate != 32000:
        value = librosa.resample(value, orig_sr=sample_rate, target_sr=32000, scale=True,
            res_type="kaiser_best" if model == "birdcode" else "soxr_hq")
    return np.ascontiguousarray(value, dtype=np.float32), 32000
