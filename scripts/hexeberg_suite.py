"""Frozen inputs and protocol for the author-recipe YOLO comparison."""
import hashlib
import json
import os
from pathlib import Path
import time

import requests

REPO = Path('/home/george-vengrovski/Documents/birdsong-detect-distill')
ROOT = Path('/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/yolo-hexeberg-25k-20260920')
OUT, ART = ROOT/'results', ROOT/'artifacts'
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
OLD = REPO/'results/paper_25000train_3seeds_10epochs_2026-09-15/manifest.json'
REVISION = '806ed2cda4ddcbe6efa194ccafff930aa0e557ce'
BIRDSET = 'https://huggingface.co/datasets/DBD-research-group/BirdSet'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def frozen(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('Frozen data changed: '+str(path))
    atomic(path,value)


def download(url, path, expected=None, size=None):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if expected and sha(path)!=expected:
            raise ValueError('Download checksum changed: '+str(path))
        return
    pending = path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            # Fresh redirect avoids expired cached HF signed URLs on lengthy transfers.
            with requests.get(url+('?download=true&attempt='+str(time.time_ns()) if 'huggingface.co' in url else ''),
                              stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                digest = hashlib.sha256(); total = 0; update = 0
                with pending.open('wb') as stream:
                    for chunk in response.iter_content(8*1024**2):
                        stream.write(chunk); digest.update(chunk); total += len(chunk)
                        if total-update >= 256*1024**2:
                            atomic(ROOT/'transfers'/f'{path.name}.json',dict(bytes=total,total=size,state='downloading'))
                            update = total
            if expected and digest.hexdigest()!=expected or size and total!=size:
                raise ValueError('Downloaded bytes differ: '+str(path))
            pending.replace(path)
            return
        except (requests.RequestException, OSError):
            if attempt==2:
                raise
            time.sleep(5*(attempt+1))


def prepare():
    from prepare_detector_study import verify
    path = OUT/'manifest.json'
    if path.exists():
        plan = read(path); verify(plan); return plan
    from ultralytics import YOLO, __version__
    if __version__!='8.3.109':
        raise ValueError('Use the authors\' Ultralytics version')
    old = read(OLD)
    protected = {str(OLD):sha(OLD)}
    authors = {}
    for variant in ['n','l']:
        released = REPO/f'models/yolo11{variant}.pt'
        saved = YOLO(released).ckpt
        if saved['version']!='8.3.109':
            raise ValueError('Unexpected author checkpoint version')
        authors[variant] = saved['train_args']
        protected[str(released)] = sha(released)
        coco = ROOT/f'models/coco_yolo11{variant}.pt'
        download(f'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{variant}.pt',coco)
        protected[str(coco)] = sha(coco)
    datasets = {k:old['datasets'][k] for k in ['25000','validation']}
    for item in datasets.values():
        protected[item['path']] = item['sha256']
    for item in old['external'].values():
        item['reference_report']=str(REPO/item['reference_report'])
        protected[item['reference_report']] = item['reference_report_sha256']
    response = requests.get(f'https://huggingface.co/api/datasets/DBD-research-group/BirdSet/tree/{REVISION}/XCL?limit=1000',timeout=30)
    response.raise_for_status()
    archives = [x for x in response.json() if x['path'].endswith('.tar.gz')]
    if len(archives)!=98:
        raise ValueError('BirdSet release inventory differs')
    plan = dict(run=ROOT.name,authors=authors,datasets=datasets,powdermill=old['powdermill'],
        external=old['external'],calibration=old['calibration'],birdset_revision=REVISION,archives=archives,
        training='COCO initialization; author checkpoint arguments; Ultralytics 8.3.109; 300 epochs maximum; batch 16; patience 50; native best-validation fitness.',
        memory='Large stores autograd saved tensors in host RAM to preserve true batch 16 on a 24GB GPU; no activation recomputation or BatchNorm change.',
        adaptation='Three seeds 0,1,2; same 25k/2.5k recording-disjoint teacher-labelled audio. Five seconds of labelled audio plus one second of digital silence; no additional unlabelled context. Teacher mel-box edges converted to Hz, then clipped to 500–12000 Hz.',
        inference='Author native waveform frontend and 6s windows/5s stride; first channel, native sample rate; confidence floor 1e-5, max_det 10000, NMS .7; no SongMAE smoothing.',
        protected_files=protected,code_sha256={str(p):sha(p) for folder in ['scripts','src'] for p in (ROOT/folder).rglob('*.py')})
    verify(plan); frozen(path,plan)
    return plan
