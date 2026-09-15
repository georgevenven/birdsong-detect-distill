#!/usr/bin/env python3
"""Stage original Hawaii soundscapes, or an explicitly marked BirdSet OGG derivative."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests
import soundfile as sf

ZENODO = "https://zenodo.org/records/7078499"
BIRDSET = "DBD-research-group/BirdSet"
REVISION = "806ed2cda4ddcbe6efa194ccafff930aa0e557ce"
MD5 = {"annotations.csv": "03b5550b6788734c5fe9728a1abc0ca2",
    "description.pdf": "b8eff939d23788b123bb85600b682842",
    "recording_location.csv": "7d6db82888c3faff3a40efa03f91ffda",
    "species.csv": "5c21df6024c41556cd334ed3d9efdf42",
    "soundscape_data.zip": "79cec7baf06770acf0ac0d519de070ca"}


def checksum(path, algorithm):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def download_ranges(url, part, connections):
    with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        if response.status_code != 206:
            return False
        total = int(response.headers["Content-Range"].split("/")[-1])
    offset = part.stat().st_size if part.exists() else 0
    if offset > total:
        raise ValueError("partial download is larger than the source")
    if offset == total:
        return True
    step = (total - offset + connections - 1) // connections
    with TemporaryDirectory(prefix=".hawaii-ranges-", dir=part.parent) as temporary:
        def fetch(start):
            end = min(total, start + step) - 1
            path = Path(temporary) / str(start)
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                    raise ValueError("server did not honor the requested byte range")
                with path.open("wb") as output:
                    for chunk in response.iter_content(1024**2):
                        output.write(chunk)
            if path.stat().st_size != end - start + 1:
                raise ValueError("incomplete byte range")
            print(f"Downloaded archive range {start:,}–{end:,}", flush=True)
            return path
        with ThreadPoolExecutor(connections) as pool:
            chunks = list(pool.map(fetch, range(offset, total, step)))
        if (part.stat().st_size if part.exists() else 0) != offset:
            raise ValueError("partial file changed during download")
        with part.open("ab") as output:
            for path in chunks:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output)
    return True


def download(url, path, algorithm, expected, connections=1):
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {path.name}", flush=True)
        if connections == 1 or not path.name.endswith(".zip") or not download_ranges(url, part, connections):
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                "--connect-timeout", "15", "--speed-time", "60", "--speed-limit", "1024",
                "--continue-at", "-", "--output", str(part), url], check=True)
        if checksum(part, algorithm) != expected:
            raise ValueError(f"checksum mismatch; partial file preserved: {part}")
        part.rename(path)
    if checksum(path, algorithm) != expected:
        raise ValueError(f"existing file checksum mismatch: {path}")
    print(f"Verified {path.name}: {path.stat().st_size:,} bytes", flush=True)
    return dict(name=path.name, url=url, bytes=path.stat().st_size, algorithm=algorithm, checksum=expected)


def extract(archive, directory):
    # Copy regular audio members only, flattening to validated basenames; never extract links.
    zipped = archive.suffix == ".zip"
    with zipfile.ZipFile(archive) if zipped else tarfile.open(archive) as source:
        entries = source.infolist() if zipped else source.getmembers()
        for entry in entries:
            name = entry.filename if zipped else entry.name
            if Path(name).suffix.lower() not in {".flac", ".ogg"} or Path(name).name.startswith("._"):
                continue
            if not zipped and not entry.isfile():
                raise ValueError(f"non-regular archive member: {name}")
            target = directory / Path(name).name
            size = entry.file_size if zipped else entry.size
            if target.exists():
                with source.open(entry) if zipped else source.extractfile(entry) as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                if target.stat().st_size != size or checksum(target, "sha256") != expected:
                    raise ValueError(f"existing extracted audio differs: {target}")
                continue
            part = target.with_suffix(target.suffix + ".part")
            with source.open(entry) if zipped else source.extractfile(entry) as stream, part.open("wb") as out:
                shutil.copyfileobj(stream, out)
            part.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("zenodo", "birdset"), default="zenodo")
    parser.add_argument("--root", type=Path, default=Path("data/hawaii"))
    parser.add_argument("--connections", type=int, choices=range(1, 5), default=1, help="Verified byte-range transfers for the large ZIP")
    args = parser.parse_args()
    root = args.root / args.source
    raw, audio = root / "raw", root / "audio"
    raw.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("completed dataset already exists; its manifest is never overwritten")
    if shutil.disk_usage(root).free < (14 if args.source == "zenodo" else 4) * 1024**3:
        raise ValueError("insufficient free space for downloads and extraction")
    files = []
    if args.source == "zenodo":
        for name, expected in MD5.items():
            files.append(download(f"{ZENODO}/files/{name}?download=1", raw / name, "md5", expected, args.connections))
    else:
        response = requests.get(f"https://huggingface.co/api/datasets/{BIRDSET}/tree/{REVISION}/UHH", timeout=30)
        response.raise_for_status()
        for item in response.json():
            name = Path(item["path"]).name
            if name != "UHH_metadata_test.parquet" and not name.startswith("UHH_test_shard_"):
                continue  # No XC training subset and no five-second repackaging.
            url = f"https://huggingface.co/datasets/{BIRDSET}/resolve/{REVISION}/{item['path']}"
            files.append(download(url, raw / name, "sha256", item["lfs"]["oid"]))
    for item in files:
        path = raw / item["name"]
        if path.name.endswith((".zip", ".tar.gz")):
            extract(path, audio)
    inventory = []
    for path in sorted(audio.iterdir()):
        info = sf.info(path)
        inventory.append(dict(filename=path.name, seconds=info.duration, sample_rate=info.samplerate,
            channels=info.channels, format=info.format, subtype=info.subtype, sha256=checksum(path, "sha256")))
    if args.source == "birdset":
        table = pd.read_parquet(raw / "UHH_metadata_test.parquet").reset_index()
        table.to_csv(root / "annotations.csv", index=False)
    else:
        table = pd.read_csv(raw / "annotations.csv").rename(columns={"Filename": "filepath", "Start Time (s)": "start_time",
            "End Time (s)": "end_time", "Low Freq (Hz)": "low_freq", "High Freq (Hz)": "high_freq", "Species eBird Code": "ebird_code"})
    if set(table.filepath) != {r["filename"] for r in inventory} or len(table) != 59583 or len(inventory) != 635:
        raise ValueError("unexpected audio / annotation recording sets or counts")
    durations = table.filepath.map({r["filename"]: r["seconds"] for r in inventory})
    warnings = {"zero_duration": int((table.end_time == table.start_time).sum()),
        "invalid_time_bounds": int(((table.start_time < 0) | (table.end_time < table.start_time) | (table.end_time > durations + .02)).sum()),
        "invalid_frequency_bounds": int(((table.low_freq < 0) | (table.high_freq <= table.low_freq) | (table.high_freq > 16000)).sum())}
    # Preserve questionable source labels verbatim; downloading is not annotation repair.
    manifest = dict(source=ZENODO, distribution=args.source, birdset_revision=REVISION if args.source == "birdset" else None,
        audio_note="BirdSet OGG/Vorbis derivative; NOT byte-identical to original FLAC" if args.source == "birdset" else "Original Zenodo FLAC release",
        recordings=len(inventory), hours=sum(r["seconds"] for r in inventory) / 3600,
        annotations=len(table), species=int(table.ebird_code.nunique()), annotation_warnings=warnings,
        annotation_sha256=checksum(root / "annotations.csv" if args.source == "birdset" else raw / "annotations.csv", "sha256"),
        files=files, audio=inventory, evaluation_run=False,
        ready_for_paper_evaluation=args.source == "zenodo" and not any(warnings.values()))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"audio", "files"}}, indent=2))


if __name__ == "__main__":
    main()
