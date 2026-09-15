#!/usr/bin/env python3
"""Remove detector positions only: three paired seeds, same 2k split and Powdermill scorer."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import run_pointwise_gelu_2k as shared
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.pointwise_no_position import ARCHITECTURE, NoPositionHead, load_heads
from prepare_detector_study import verify
from run_new_teacher_2k import OUT as PARENT, ARTIFACTS as PARENT_ARTIFACTS

RUN = 'pointwise_no_position_2000train_800val_2026-09-14'
OUT = PARENT.parent / RUN
ARTIFACTS = PARENT_ARTIFACTS.parent / RUN
GELU_OUT = shared.OUT
SCRIPT = 'scripts/run_pointwise_no_position_2k.py'
MODEL = 'src/birdsong_detect_distill/pointwise_no_position.py'


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        verify(plan)
        return plan
    parent = json.loads((PARENT / 'manifest.json').read_text())
    verify(parent)
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('linear controls must be complete')
    protected, controls = dict(parent['protected_files']), []
    for seed in parent['seeds']:
        path = PARENT / 'conditions' / f'layers0_d128_new2k_s{seed}.json'
        spec = json.loads(path.read_text())
        shared.study.measure(spec,parent)
        controls.append({**spec,'reused':True,'positions':True})
        for p in [path,Path(spec['checkpoint']),Path(spec['report'])]:
            protected[str(p)] = digest(p)
    for p in [PARENT / 'manifest.json',PARENT / 'status.json',PARENT / 'pointwise/summary.json']:
        protected[str(p)] = digest(p)
    code = {**parent['code_sha256'], **{p:digest(p) for p in
        [SCRIPT,MODEL,'scripts/run_pointwise_gelu_2k.py','src/birdsong_detect_distill/pointwise_gelu.py']}}
    plan = {**parent,'run':RUN,'layers':[0],'controls':controls,'artifacts':str(ARTIFACTS),
        'protected_files':protected,'code_sha256':code,
        'architecture':'LayerNorm(768) -> Linear(768,128) -> Linear(128,32); no detector positional embeddings or GELU',
        'ablation':'Remove learned detector frequency/time embeddings only; frozen backbone positions and LayerNorm remain',
        'pairing':'Same initialization before deleting positions, exact shared parameter values and sample/CPU RNG histories',
        'head_parameters':104096,'control_head_parameters':232608,
        'schedule':'Wait for existing GELU service to exit; use Twins only; leave V100 running and Twins Qwen paused'}
    verify(plan)
    write_json(OUT / 'manifest.json',plan)
    return plan


def check_pair(saved, control, audit):
    old = torch.load(control['checkpoint'],map_location='cpu',weights_only=True)
    if audit['control_initial_sha256'] != old['initial_head_sha256'] or audit['no_position_initial_sha256'] != saved['initial_head_sha256']:
        raise ValueError('initial parameters do not match the paired control')
    expected = {k for k in old['head'] if k not in ['time','frequency']}
    if set(saved['head']) != expected or sum(v.numel() for v in saved['head'].values()) != 104096:
        raise ValueError('unexpected head parameter removal')
    if len(saved['training_history']) != 5 or len(old['training_history']) != 5:
        raise ValueError('five matched epochs required')
    for a,b in zip(saved['training_history'],old['training_history']):
        if any(a[k]!=b[k] for k in ['epoch','sample_order_sha256','cpu_rng_sha256']):
            raise ValueError('paired training order or randomness differs')


def train(spec, plan):
    target = Path(spec['checkpoint'])
    pending, audit_path = target.with_suffix('.training.pt'), target.with_suffix('.initialization.json')
    if target.exists():
        raise ValueError('do not overwrite a completed checkpoint')
    if not pending.exists():
        def factory(*args, **kwargs):
            head = NoPositionHead(*args,**kwargs)
            audit = dict(control_initial_sha256=head.control_initial_sha256,
                no_position_initial_sha256=hashlib.sha256(b''.join(
                    v.detach().cpu().numpy().tobytes() for v in head.state_dict().values())).hexdigest())
            shared.study.frozen_json(audit_path,audit)
            return head
        shared.trainer.DenseHead = factory
        sys.argv = ['scripts/train.py','--annotations',plan['datasets']['new2k']['path'],
            '--validation-annotations',plan['datasets']['validation']['path'],'--backbone',plan['backbone'],
            '--backbone-revision',plan['revision'],'--hidden','128','--head-layers','0',
            '--seed',str(spec['seed']),'--epochs','5','--learning-rate','0.001','--weight-decay','0.0001',
            '--batch-size','16','--eval-batch-size','4','--accumulation','1','--dropout','0.1',
            '--tv-weight','0','--no-target-smoothing','--log-every','5','--out',str(pending)]
        shared.trainer.main()
    saved = torch.load(pending,map_location='cpu',weights_only=True)
    audit = json.loads(audit_path.read_text())
    control = next(c for c in plan['controls'] if c['seed']==spec['seed'])
    check_pair(saved,control,audit)
    NoPositionHead(768,128,4,1000,32,1,layers=0).load_state_dict(saved['head'])
    saved.update(detector_architecture=ARCHITECTURE,head_activation='none',detector_position_embeddings=False,
        layer_norm=True,initialization_audit=audit,experiment_code_sha256=plan['code_sha256'],
        paired_control_sha256=digest(control['checkpoint']))
    staged = target.with_suffix('.tagged.pt')
    torch.save(saved,staged)
    staged.replace(target)
    print(f'Validated and tagged no-position checkpoint: {target}',flush=True)


def evaluate(path, spec, plan):
    saved = torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
    if saved.get('detector_architecture') != ARCHITECTURE or saved.get('experiment_code_sha256') != plan['code_sha256']:
        raise ValueError('incorrect architecture or experiment provenance')
    check_pair(saved,next(c for c in plan['controls'] if c['seed']==spec['seed']),saved['initialization_audit'])
    shared.evaluator.OUT, shared.scorer.load_heads = OUT, load_heads
    shared.evaluator.evaluate(path)
    shared.study.measure(spec,plan)


def run(plan):
    unit = 'birdsong-detector-gelu2k-20260914.service'
    write_json(OUT / 'status.json',dict(state='waiting_for_gelu',updated_unix=time.time()))
    while subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip() in ['active','activating','deactivating']:
        time.sleep(10)
    if json.loads((GELU_OUT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('GELU did not finish successfully; inspect it before new GPU work')
    for suffix in ['twins','server']:
        state = subprocess.check_output(['systemctl','--user','show',
            f'birdsong-qwen-xc50k-{suffix}-20260914.service','-p','ActiveState','--value'],text=True).strip()
        if state != 'inactive':
            raise ValueError('Twins Qwen must remain paused')
    for gpu in range(2):
        if torch.cuda.mem_get_info(gpu)[0] < 20*1024**3:
            raise ValueError('Twins GPU occupied; preserve other work')
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    shared.OUT, shared.ARTIFACTS, shared.SCRIPT = OUT, ARTIFACTS, SCRIPT
    shared.study.OUT, shared.study.ARTIFACTS = OUT, ARTIFACTS
    specs = [{**shared.study.condition('no_position',128,seed,dataset='new2k',layers=0),'positions':False}
        for seed in plan['seeds']]
    write_json(OUT / 'status.json',dict(state='running',new_runs=3,reused_controls=3,started_unix=time.time()))
    def lane(gpu):
        for spec in specs[gpu::2]:
            shared.execute(spec,plan,gpu)
        write_json(OUT / f'gpu{gpu}.json',dict(state='complete',updated_unix=time.time()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane,gpu) for gpu in range(2)]
        for future in futures:
            future.result()
    groups = {}
    for name,conditions in [('with_position',plan['controls']),('no_position',specs)]:
        values = [shared.study.measure(spec,plan) for spec in conditions]
        groups[name] = dict(per_seed=values,**{f'{metric}_{stat}':float(fn([v[metric] for v in values]))
            for metric in ['ap','iou'] for stat,fn in [('mean',np.mean),('sd',lambda x:np.std(x,ddof=1))]})
    deltas = [dict(seed=a['seed'],**{k:float(b[k]-a[k]) for k in ['ap','iou']}) for a,b in
        zip(groups['with_position']['per_seed'],groups['no_position']['per_seed'])]
    verify(plan)
    write_json(OUT / 'summary.json',dict(groups=groups,paired_no_position_minus_control=deltas,
        manifest_sha256=digest(OUT / 'manifest.json'),uncertainty='sample SD over seeds, not dataset confidence intervals'))
    write_json(OUT / 'status.json',dict(state='complete',new_runs=3,reused_controls=3,completed_unix=time.time()))
    print(json.dumps({k:{f:v[f] for f in ['ap_mean','ap_sd','iou_mean','iou_sd']} for k,v in groups.items()},indent=2),flush=True)


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--train',type=Path)
    mode.add_argument('--evaluate',type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    child = args.train or args.evaluate
    if not child:
        lock = (OUT / 'driver.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = prepare()
        if child:
            spec = json.loads(child.read_text())
            train(spec,plan) if args.train else evaluate(child,spec,plan)
        elif args.prepare:
            print('Frozen: three no-position runs; 2,000 s train + 800 s validation; existing linear controls.')
        else:
            run(plan)
    except Exception as error:
        if not child:
            write_json(OUT / 'status.json',dict(state='failed',error=str(error),updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
