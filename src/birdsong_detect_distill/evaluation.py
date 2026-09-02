import io
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc


WABAD_SITES = """ARD BAM BIAL BMT BOLIN BRCAS BRE BUR CARI CAT CB CLH COU CRUZ DEVA DONG DUNAS DYOM EMP EVROS FEU FNCA GLEN GTLU HAG HAK HAR HONDO HUAP JUNCA KAR KIB LIM MABI MAPIMI MARTI MILLAN MONTEB MOPU NAV NL OESF OIO OLIV PETI PGF PINA PITI POZO PUUL QR RBA RFP RGU RME ROTOK SAL SBN SCHF SCHG SD SITH SLOB SPMCO TAM UNI VER VIL""".split()
THRESHOLDS = np.linspace(0, 1, 101)


def audio(zip_file, member):
    encoded = zip_file.read(member)
    try:
        waveform, sample_rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "32000", "pipe:1"],
            input=encoded, capture_output=True, check=True).stdout
        return np.frombuffer(decoded, "<f4")
    waveform = waveform.mean(1)
    return librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000, scale=True) if sample_rate != 32000 else waveform


def wabad(root, sites=WABAD_SITES):
    annotations = pd.read_csv(root / "wabad" / "Pooled annotations.csv")
    columns = ["Begin_Time_(s)", "End_Time_(s)", "Low_Freq_(Hz)", "High_Freq_(Hz)"]
    for site in sites:
        with zipfile.ZipFile(root / "wabad" / f"{site}.zip") as archive:
            members = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".wav")}
            tables = {name: events[columns].to_numpy(float)
                for name, events in annotations[annotations.Site == site].groupby("Recording")}
            for name, member in members.items():
                yield f"{site}/{name}", audio(archive, member), tables.get(name, np.empty((0, 4)))


def powdermill(root):
    with zipfile.ZipFile(root / "powdermill" / "wav_Files.zip") as sounds, zipfile.ZipFile(root / "powdermill" / "annotation_Files.zip") as labels:
        members = {Path(name).stem: name for name in sounds.namelist() if name.lower().endswith(".wav")}
        for member in labels.namelist():
            if member.lower().endswith(".txt"):
                table = pd.read_csv(labels.open(member), sep="\t")
                stem = Path(member).name.split(".Table")[0]
                columns = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
                yield stem, audio(sounds, members[stem]), table[columns].to_numpy(float)


def xcsl(root):
    with zipfile.ZipFile(root / "xcaj" / "Audio.zip") as sounds, zipfile.ZipFile(root / "xcaj" / "Annotation.zip") as labels:
        members = {Path(name).stem: name for name in sounds.namelist() if Path(name).suffix.lower() in (".mp3", ".wav")}
        for member in labels.namelist():
            if not member.lower().endswith(".svl"):
                continue
            tree = ET.fromstring(labels.read(member))
            sample_rate = float(tree.find("./data/model").attrib["sampleRate"])
            events = []
            for point in tree.findall("./data/dataset/point"):
                low = float(point.attrib["value"])
                events.append([float(point.attrib["frame"]) / sample_rate,
                    (float(point.attrib["frame"]) + float(point.attrib["duration"])) / sample_rate,
                    low, low + float(point.attrib["extent"])])
            stem = Path(member).stem
            yield stem, audio(sounds, members[stem]), np.asarray(events)


def counters():
    return np.zeros((len(THRESHOLDS), 3), np.int64), {
        iou: np.zeros((len(THRESHOLDS), 3), np.int64) for iou in (.2, .5)}


def spans(mask, rate):
    change = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return np.column_stack((np.flatnonzero(change == 1), np.flatnonzero(change == -1))) / rate


def events(score, rate, threshold):
    found, merged = spans(score >= threshold, rate), []
    for onset, offset in found:
        if merged and onset - merged[-1][1] < 1:
            merged[-1][1] = offset
        elif offset - onset >= .01:
            merged.append([onset, offset])
    return np.asarray(merged).reshape(-1, 2)


def matches(reference, estimate, threshold):
    if not len(reference) or not len(estimate):
        return np.asarray([0, len(estimate), len(reference)])
    intersection = np.maximum(0, np.minimum(reference[:, None, 1], estimate[None, :, 1])
        - np.maximum(reference[:, None, 0], estimate[None, :, 0]))
    union = np.maximum(reference[:, None, 1], estimate[None, :, 1]) - np.minimum(reference[:, None, 0], estimate[None, :, 0])
    valid = intersection / np.maximum(union, 1e-12) > threshold
    row, column = linear_sum_assignment(valid, maximize=True)
    true = int(valid[row, column].sum())
    return np.asarray([true, len(estimate) - true, len(reference) - true])


def resample_max(score, rate, target_rate=100):
    output = np.zeros(int(len(score) / rate * target_rate), np.float32)
    for index, value in enumerate(score):
        left, right = int(index / rate * target_rate), int(np.ceil((index + 1) / rate * target_rate))
        output[left:min(right, len(output))] = np.maximum(output[left:min(right, len(output))], value)
    return output


def score_recording(score, rate, reference, duration, frame_counts, event_counts):
    score_100 = resample_max(score, rate)
    truth = np.zeros(len(score_100), bool)
    for onset, offset in reference:
        truth[max(0, int(onset * 100)):min(len(truth), int(np.ceil(offset * 100)))] = True
    prediction = score_100[None] >= THRESHOLDS[:, None]
    frame_counts[:, 0] += (prediction & truth).sum(1)
    frame_counts[:, 1] += (prediction & ~truth).sum(1)
    frame_counts[:, 2] += (~prediction & truth).sum(1)
    reference = reference[(reference[:, 0] < duration) & (reference[:, 1] > 0)].copy()
    reference[:, 0], reference[:, 1] = np.maximum(reference[:, 0], 0), np.minimum(reference[:, 1], duration)
    for index, threshold in enumerate(THRESHOLDS):
        estimate = events(score, rate, threshold)
        for iou in event_counts:
            event_counts[iou][index] += matches(reference, estimate, iou)


def average_precision(counts):
    tp, fp, fn = counts.T
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=tp + fp != 0)
    recall = np.divide(tp, tp + fn, out=np.ones_like(tp, dtype=float), where=tp + fn != 0)
    order = np.argsort(np.r_[0, recall])
    recall, precision = np.r_[0, recall][order], np.r_[1, precision][order]
    return float(auc(recall, np.maximum.accumulate(precision[::-1])[::-1]))


def metrics(frame, event):
    return {"frame_ap": average_precision(frame), "event_ap_iou_0.2": average_precision(event[.2]),
        "event_ap_iou_0.5": average_precision(event[.5])}
