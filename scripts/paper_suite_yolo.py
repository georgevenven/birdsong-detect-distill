#!/usr/bin/env python3
"""Train both YOLO sizes on identical frozen teacher masks, retaining native validation."""
import argparse
import csv
import fcntl
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

import paper_suite as suite
from birdsong_detect_distill.benchmark_data import digest
from birdsong_detect_distill.birdbox import model_path
from birdsong_detect_distill.data import read_rows
from prepare_yolo import write_split
from prepare_detector_study import verify


def prepare_data(plan):
    root = suite.ART/'yolo_dataset'
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'prepare.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    manifest_sha = digest(suite.OUT/'manifest.json')
    marker = root/'summary.json'
    if marker.exists():
        summary = json.loads(marker.read_text())
        if summary['manifest_sha256']!=manifest_sha or digest(root/'dataset.yaml')!=summary['yaml_sha256']:
            raise ValueError('YOLO dataset provenance changed')
        return root/'dataset.yaml',summary
    datasets = {key:plan['datasets'][source] for key,source in [('train','15000'),('val','validation')]}
    summary = {key:write_split(read_rows(data['path'],0),key,root,suite.ROOT/'data/xcl/shards')
        for key,data in datasets.items()}
    yaml = root/'dataset.yaml'
    yaml.write_text(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n')
    summary.update(manifest_sha256=manifest_sha,annotations_sha256=datasets['train']['sha256'],
        validation_annotations_sha256=datasets['val']['sha256'],training_recordings=datasets['train']['recording_ids'],
        validation_recordings=datasets['val']['recording_ids'],training_seconds=datasets['train']['seconds'],
        validation_seconds=datasets['val']['seconds'],train_all=False,yaml_sha256=digest(yaml),
        image_representation='5s_128_mel_viridis_2048x512',labels='same integer mel/time boxes as SongMAE; no axes overlay')
    suite.atomic(marker,summary)
    return yaml,summary


def run(variant):
    plan = suite.prepare()
    out = suite.ART/'yolo'/variant
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'driver.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint = out/'model.pt'
    if checkpoint.exists() and checkpoint.with_suffix('.json').exists():
        sidecar = json.loads(checkpoint.with_suffix('.json').read_text())
        if digest(checkpoint)!=sidecar['sha256'] or sidecar['manifest_sha256']!=digest(suite.OUT/'manifest.json'):
            raise ValueError('YOLO completed checkpoint changed')
        return
    if torch.cuda.mem_get_info(0)[0]<20*1024**3:
        raise ValueError('GPU occupied')
    data, summary = prepare_data(plan)
    source = model_path('yolo11'+variant,suite.ROOT/'models')
    args_path = out/'fit/args.yaml'
    last = out/'fit/weights/last.pt'
    best = out/'fit/weights/best.pt'
    history_path = out/'fit/results.csv'
    history = [{k.strip():v for k,v in row.items()} for row in csv.DictReader(history_path.open())] if history_path.exists() else []
    finished = bool(history) and int(float(history[-1]['epoch']))>=50 and best.exists()
    if not finished:
        if last.exists():
            model = YOLO(last)
            model.train(resume=True,device=0,workers=4)
        else:
            model = YOLO(source)
            # A common explicit batch avoids architecture-dependent optimizer auto-selection.
            model.train(data=str(data),epochs=50,imgsz=1024,batch=8,device=0,workers=4,
                project=str(out),name='fit',exist_ok=True,patience=0,seed=0,deterministic=True,
                plots=False,val=True,cache=False,save=True,save_period=-1)
        history = list(csv.DictReader(history_path.open()))
    history = [{k.strip():v for k,v in row.items()} for row in history]
    if len(history)!=50 or not best.exists():
        raise ValueError('incomplete YOLO training history')
    selected = max(reversed(history),key=lambda r:float(r['metrics/mAP50-95(B)']))
    temporary = checkpoint.with_suffix('.pending.pt')
    shutil.copy2(best,temporary)
    temporary.replace(checkpoint)
    suite.atomic(checkpoint.with_suffix('.json'),dict(dataset=summary,epochs=50,batch_size=8,
        initialization='released_birdbox_yolo11'+variant,initial_checkpoint_sha256=digest(source),
        selection='best_validation_box_map',selected_epoch=int(float(selected['epoch'])),
        validation_selection_metric='native_box_mAP50-95',training_log_sha256=digest(history_path),
        training_args=str(args_path),training_args_sha256=digest(args_path),checkpoint=str(checkpoint),
        sha256=digest(checkpoint),manifest_sha256=digest(suite.OUT/'manifest.json')))
    verify(plan)
    print('Completed:',checkpoint,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['n','l'],required=True)
    run(parser.parse_args().variant)
