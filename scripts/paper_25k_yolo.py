#!/usr/bin/env python3
"""Matched teacher-labelled YOLO budgets, ten epochs and three training seeds."""
import argparse
import csv
import fcntl
import json
import shutil

import torch
from ultralytics import YOLO

import paper_25k_suite as suite
from birdsong_detect_distill.benchmark_data import digest
from birdsong_detect_distill.birdbox import model_path
from birdsong_detect_distill.data import read_rows
from prepare_detector_study import verify
from prepare_yolo import write_split


def dataset(plan, budget):
    root = suite.ART/'yolo_dataset'/str(budget)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'prepare.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
    marker, yaml = root/'summary.json', root/'dataset.yaml'
    sha = digest(suite.OUT/'manifest.json')
    if marker.exists():
        summary = json.loads(marker.read_text())
        if summary['manifest_sha256'] != sha or summary['yaml_sha256'] != digest(yaml):
            raise ValueError('rendered dataset provenance differs')
        return yaml, summary
    splits = {'train':plan['datasets'][str(budget)], 'val':plan['datasets']['validation']}
    summary = {key:write_split(read_rows(data['path'],0),key,root,suite.ROOT/'data/xcl/shards')
               for key,data in splits.items()}
    yaml.write_text(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n')
    summary.update(manifest_sha256=sha, yaml_sha256=digest(yaml),
        annotations_sha256=splits['train']['sha256'], validation_annotations_sha256=splits['val']['sha256'],
        training_seconds=budget, validation_seconds=2500, train_all=False,
        training_recordings=splits['train']['recording_ids'], validation_recordings=splits['val']['recording_ids'],
        image_representation='5s_128_mel_viridis_2048x512', labels='same integer mel/time boxes as SongMAE, no axes overlay')
    if summary['train']['windows'] != budget//5 or summary['val']['windows'] != 500:
        raise ValueError('YOLO budget differs')
    suite.atomic(marker,summary)
    return yaml, summary


def run(job):
    plan = suite.prepare(); family, variant, budget, seed = suite.job_parts(job)
    if family != 'yolo' or job not in suite.YOLO_JOBS:
        raise ValueError('unknown YOLO job')
    out, reports = suite.ART/'yolo'/job, suite.OUT/'runs'/job
    out.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    lock = (out/'driver.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint = out/'model.pt'; sidecar = checkpoint.with_suffix('.json')
    manifest_sha = digest(suite.OUT/'manifest.json')
    if checkpoint.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if saved['sha256'] != digest(checkpoint) or saved['manifest_sha256'] != manifest_sha or saved['seed'] != seed:
            raise ValueError('completed YOLO checkpoint changed')
        return
    if torch.cuda.mem_get_info(0)[0] < 20*1024**3:
        raise ValueError('GPU occupied')
    data, summary = dataset(plan,budget)
    source = model_path('yolo11'+variant,suite.ROOT/'models')
    last, best = out/'fit/weights/last.pt', out/'fit/weights/best.pt'
    history_path, args_path = out/'fit/results.csv', out/'fit/args.yaml'
    def history():
        if not history_path.exists():
            return []
        with history_path.open() as stream:
            return [{k.strip():v for k,v in r.items()} for r in csv.DictReader(stream)]
    rows = history()
    finished = rows and int(float(rows[-1]['epoch'])) >= plan['epochs'] and best.exists()
    suite.atomic(reports/'status.json',dict(state='training',job=job,seed=seed,epochs=plan['epochs']))
    if not finished:
        model = YOLO(last if last.exists() else source)
        if last.exists():
            model.train(resume=True,device=0,workers=4)
        else:
            model.train(data=str(data),epochs=plan['epochs'],imgsz=1024,batch=8,device=0,workers=4,
                project=str(out),name='fit',exist_ok=True,patience=0,seed=seed,deterministic=True,
                plots=False,val=True,cache=False,save=True,save_period=-1)
        rows = history()
    if len(rows) != plan['epochs'] or not best.exists():
        raise ValueError('incomplete YOLO training')
    selected = max(reversed(rows),key=lambda r:float(r['metrics/mAP50-95(B)']))
    pending = checkpoint.with_suffix('.pending.pt'); shutil.copy2(best,pending); pending.replace(checkpoint)
    metadata = dict(dataset=summary,epochs=plan['epochs'],batch_size=8,seed=seed,job=job,
        initialization='released_birdbox_yolo11'+variant,initial_checkpoint_sha256=digest(source),
        selection='best_validation_box_map',selected_epoch=int(float(selected['epoch'])),
        validation_selection_metric='native_box_mAP50-95',training_log_sha256=digest(history_path),
        training_args=str(args_path),training_args_sha256=digest(args_path),checkpoint=str(checkpoint),
        sha256=digest(checkpoint),manifest_sha256=manifest_sha)
    suite.atomic(sidecar,metadata); suite.atomic(reports/'checkpoint.json',metadata)
    suite.atomic(reports/'status.json',dict(state='trained',job=job,seed=seed,epochs=plan['epochs']))
    verify(plan)
    print('Completed:',checkpoint,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',choices=suite.YOLO_JOBS,required=True)
    run(parser.parse_args().job)
