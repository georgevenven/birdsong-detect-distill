#!/usr/bin/env python3
"""Two detached GPU lanes: native author-recipe YOLO, then frozen external scoring."""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import hexeberg_suite as suite

JOBS=[f'yolo_{variant}_25000_s{seed}' for variant in ['n','l'] for seed in range(3)]


def train(job):
    import torch
    from ultralytics import YOLO
    plan=suite.prepare(); variant=job.split('_')[1]; seed=int(job[-1])
    out=suite.ART/job; report=suite.OUT/'runs'/job
    out.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    if (report/'checkpoint.json').exists():
        saved=suite.read(report/'checkpoint.json')
        if suite.sha(saved['path'])!=saved['sha256']:
            raise ValueError('Completed checkpoint changed')
        return
    if torch.cuda.mem_get_info(0)[0]<22*1024**3:
        raise ValueError('GPU occupied; do not compete with another task')
    coco=suite.ROOT/f'models/coco_yolo11{variant}.pt'
    last=out/'fit/weights/last.pt'
    args=dict(plan['authors'][variant])
    args.update(model=str(coco),data=str(suite.ROOT/'dataset/dataset.yaml'),seed=seed,device=0,
        workers=4,project=str(out),name='fit',exist_ok=True,plots=False)
    suite.frozen(report/'requested_train_args.json',args)
    suite.atomic(report/'status.json',dict(state='training',seed=seed,updated_unix=time.time()))
    model=YOLO(last if last.exists() else coco)
    with torch.autograd.graph.save_on_cpu(pin_memory=True) if variant=='l' else nullcontext():
        model.train(resume=True,device=0,workers=4) if last.exists() else model.train(**args)
    best=out/'fit/weights/best.pt'
    if not best.exists():
        raise ValueError('Training did not produce best.pt')
    destination=out/'model.pt'
    shutil.copy2(best,destination)
    suite.atomic(report/'checkpoint.json',dict(path=str(destination),sha256=suite.sha(destination),seed=seed,
        selection='Ultralytics 8.3.109 native validation fitness, best.pt',
        manifest_sha256=suite.sha(suite.OUT/'manifest.json'),train_args_sha256=suite.sha(out/'fit/args.yaml'),
        history_sha256=suite.sha(out/'fit/results.csv'),initialization_sha256=suite.sha(coco)))
    suite.atomic(report/'status.json',dict(state='trained',updated_unix=time.time()))


def evaluate(job,dataset):
    import torch
    from ultralytics import YOLO
    import paper_suite_evaluate as shared
    from birdsong_detect_distill.birdbox import predict_boxes
    from summarize_powdermill_loro import checked_rows,evaluate as cross_calibrate
    shared.suite=SimpleNamespace(ROOT=suite.REPO,RAW=suite.RAW,ART=suite.ART,OUT=suite.OUT,
        prepare=suite.prepare,frozen=suite.frozen,atomic=suite.atomic)
    def identity(name,plan):
        record=suite.read(suite.OUT/'runs'/name/'checkpoint.json'); checkpoint=Path(record['path'])
        if suite.sha(checkpoint)!=record['sha256'] or record['manifest_sha256']!=suite.sha(suite.OUT/'manifest.json'):
            raise ValueError('Evaluation checkpoint provenance differs')
        model=YOLO(checkpoint)
        metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=record['sha256'],training=record,
            input_representation='BirdBox native 6s linear-Hz magma',channel='first',sample_rate='native',
            window_seconds=6,stride_seconds=5,confidence_floor=1e-5,nms_iou=.7,max_det=10000,
            ultralytics='8.3.109',smoothing='none',adapter_sha256=suite.sha(__file__))
        @torch.inference_mode()
        def predict(audio,rate):
            return dict(boxes=predict_boxes(model,audio,rate,'cuda:0',confidence=1e-5,batch_size=8,max_det=10000))
        return 'birdbox',metadata,predict
    shared.identity=identity
    shared.run(job,dataset)
    if dataset=='powdermill':
        path=suite.OUT/'external'/job/'powdermill.json'; report=suite.read(path)
        rows=[dict(r,area={'full':r['area']['full']}) for r in checked_rows(report)]
        spec=dict(id=job,family=job.rsplit('_s',1)[0],label=job,seed=int(job[-1]),threshold_floor_index=0,path=path)
        suite.atomic(path.with_name('loro.json'),cross_calibrate(spec,dict(report,summary=report['summary']['full']),rows))


def lane(gpu,variant):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)}
    for seed in range(3):
        job=f'yolo_{variant}_25000_s{seed}'
        completed=suite.OUT/'runs'/job/'status.json'
        if completed.exists() and suite.read(completed)['state']=='complete':
            continue
        with (suite.OUT/'logs'/f'{job}.log').open('a') as log:
            for mode in ['train','powdermill','wabad','hawaii','xcsl','nips4bplus']:
                print(job,mode,flush=True)
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'--job',job,'--mode',mode],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        suite.atomic(completed,dict(state='complete',datasets=['powdermill','wabad','hawaii','xcsl','nips4bplus'],updated_unix=time.time()))


def driver():
    suite.OUT.mkdir(parents=True,exist_ok=True); (suite.OUT/'logs').mkdir(exist_ok=True)
    lock=(suite.OUT/'driver.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        suite.atomic(suite.OUT/'status.json',dict(state='preparing_audio',updated_unix=time.time()))
        suite.prepare()
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('prepare_hexeberg_audio.py'))],check=True)
        suite.atomic(suite.OUT/'status.json',dict(state='training_and_evaluating',updated_unix=time.time()))
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(lane,0,'n'),pool.submit(lane,1,'l')]
            for future in futures:
                future.result()
        rows=[]
        for job in JOBS:
            for dataset in ['wabad','hawaii','xcsl','nips4bplus']:
                report=suite.read(suite.OUT/'external'/job/f'{dataset}.json')
                metric='full' if dataset in ['wabad','hawaii'] else 'temporal'
                score=report['site_macro' if dataset=='wabad' else 'summary'][metric]
                rows.append(dict(job=job,dataset=dataset,ap=score['ap'],iou=score['iou']))
        suite.atomic(suite.OUT/'external_summary.json',rows)
        suite.atomic(suite.OUT/'status.json',dict(state='complete',jobs=6,updated_unix=time.time()))
    except BaseException as error:
        suite.atomic(suite.OUT/'status.json',dict(state='failed',error=repr(error),updated_unix=time.time()))
        raise


if __name__=='__main__':
    if os.uname().nodename!='Lambda-Twins':
        raise SystemExit('Twins only')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',choices=JOBS)
    parser.add_argument('--mode',choices=['train','powdermill','wabad','hawaii','xcsl','nips4bplus'])
    args=parser.parse_args()
    if args.job:
        train(args.job) if args.mode=='train' else evaluate(args.job,args.mode)
    else:
        driver()
