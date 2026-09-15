#!/usr/bin/env python3
"""Three paired GELU runs; reuse the completed linear controls and frozen Powdermill scorer."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import evaluate_current_backbones as scorer
import evaluate_detector_study as evaluator
import run_detector_study as study
import train as trainer
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.pointwise_gelu import ARCHITECTURE, GeluPointwiseHead, load_heads
from prepare_detector_study import verify
from run_new_teacher_2k import OUT as PARENT, ARTIFACTS as PARENT_ARTIFACTS

RUN = 'pointwise_gelu_2000train_800val_2026-09-14'
OUT = PARENT.parent / RUN
ARTIFACTS = PARENT_ARTIFACTS.parent / RUN
SCRIPT = 'scripts/run_pointwise_gelu_2k.py'
MODEL = 'src/birdsong_detect_distill/pointwise_gelu.py'


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        verify(plan)
        return plan
    parent = json.loads((PARENT / 'manifest.json').read_text())
    verify(parent)
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('parent comparison must be complete')
    controls = []
    protected = dict(parent['protected_files'])
    for seed in parent['seeds']:
        source = PARENT / 'conditions' / f'layers0_d128_new2k_s{seed}.json'
        spec = json.loads(source.read_text())
        study.measure(spec, parent)
        controls.append({**spec, 'reused':True, 'activation':'none'})
        for p in [source, Path(spec['checkpoint']), Path(spec['report'])]:
            protected[str(p)] = digest(p)
    for p in [PARENT / 'manifest.json', PARENT / 'status.json', PARENT / 'pointwise/summary.json']:
        protected[str(p)] = digest(p)
    plan = {**parent, 'run':RUN, 'layers':[0], 'activations':['none','gelu'], 'controls':controls,
        'artifacts':str(ARTIFACTS), 'protected_files':protected,
        'code_sha256':{**parent['code_sha256'], SCRIPT:digest(SCRIPT), MODEL:digest(MODEL)},
        'architecture':'LayerNorm -> Linear(768,128) -> add time/frequency embeddings -> GELU -> Linear(128,32)',
        'ablation':'Only add exact GELU after positional addition and before output; no dropout or extra parameters',
        'implementation':'Isolated head factory injected into trainer/scorer in child processes; frozen files unchanged',
        'pairing':'Same initial parameter values, sample order, CPU RNG and data; three existing controls reused'}
    verify(plan)
    write_json(path, plan)
    return plan


def check_pair(saved, control):
    old = torch.load(control['checkpoint'], map_location='cpu', weights_only=True)
    if saved['initial_head_sha256'] != old['initial_head_sha256']:
        raise ValueError('initial shared parameter values differ')
    for a,b in zip(saved['training_history'],old['training_history']):
        if any(a[k] != b[k] for k in ['epoch','sample_order_sha256','cpu_rng_sha256']):
            raise ValueError('paired training order or randomness differs')
    if len(saved['training_history']) != 5 or len(old['training_history']) != 5:
        raise ValueError('not five matched training epochs')
    if sum(v.numel() for v in saved['head'].values()) != sum(v.numel() for v in old['head'].values()):
        raise ValueError('GELU must not add parameters')


def train(spec, plan):
    target = Path(spec['checkpoint'])
    if target.exists():
        raise ValueError('do not overwrite a completed checkpoint')
    pending = target.with_suffix('.training.pt')
    control = next(c for c in plan['controls'] if c['seed']==spec['seed'])
    if not pending.exists():
        trainer.DenseHead = GeluPointwiseHead
        sys.argv = ['scripts/train.py', '--annotations',plan['datasets']['new2k']['path'],
            '--validation-annotations',plan['datasets']['validation']['path'], '--backbone',plan['backbone'],
            '--backbone-revision',plan['revision'], '--hidden','128', '--head-layers','0',
            '--seed',str(spec['seed']), '--epochs','5', '--learning-rate','0.001', '--weight-decay','0.0001',
            '--batch-size','16', '--eval-batch-size','4', '--accumulation','1', '--dropout','0.1',
            '--tv-weight','0', '--no-target-smoothing', '--log-every','5', '--out',str(pending)]
        trainer.main()
    saved = torch.load(pending, map_location='cpu', weights_only=True)
    head = GeluPointwiseHead(768,128,4,1000,32,1,layers=0)
    head.load_state_dict(saved['head'])
    check_pair(saved, control)
    saved.update(detector_architecture=ARCHITECTURE, head_activation='gelu',
        activation_position='after projection and positional embeddings, before output linear',
        experiment_code_sha256=plan['code_sha256'], paired_control_sha256=digest(control['checkpoint']))
    staged = target.with_suffix('.tagged.pt')
    torch.save(saved, staged)
    staged.replace(target)
    print(f'Validated and tagged GELU checkpoint: {target}',flush=True)


def evaluate(path, spec, plan):
    saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
    if saved.get('detector_architecture') != ARCHITECTURE or saved.get('experiment_code_sha256') != plan['code_sha256']:
        raise ValueError('incorrect activation or experiment provenance')
    check_pair(saved, next(c for c in plan['controls'] if c['seed']==spec['seed']))
    # Local to this process; no shared source or old loaders are rewritten.
    evaluator.OUT, scorer.load_heads = OUT, load_heads
    evaluator.evaluate(path)
    study.measure(spec, plan)


def execute(spec, plan, gpu):
    path = OUT / 'conditions' / f'{spec["name"]}.json'
    study.frozen_json(path, spec)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'OMP_NUM_THREADS':'4',
        'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'HF_HUB_OFFLINE':'1'}
    actions = ['evaluate'] if Path(spec['checkpoint']).exists() else ['train','evaluate']
    for action in actions:
        verify(plan)
        if shutil.disk_usage(ARTIFACTS).free < 2*1024**3 or shutil.disk_usage('.').free < 1024**3:
            raise ValueError('low disk space; do not delete existing data')
        write_json(OUT / f'gpu{gpu}.json',dict(state=action, condition=spec['name'], updated_unix=time.time()))
        print(f'{action}: {spec["name"]} on Twins GPU {gpu}',flush=True)
        with (OUT / 'logs' / f'{action}_{spec["name"]}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',SCRIPT,f'--{action}',str(path)],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)


def run(plan):
    for suffix in ['twins','server']:
        state = subprocess.check_output(['systemctl','--user','show',
            f'birdsong-qwen-xc50k-{suffix}-20260914.service','-p','ActiveState','--value'],text=True).strip()
        if state != 'inactive':
            raise ValueError('Twins Qwen must remain paused')
    for gpu in range(2):
        if torch.cuda.mem_get_info(gpu)[0] < 20*1024**3:
            raise ValueError('Twins GPU is occupied; preserve other work')
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    study.OUT, study.ARTIFACTS = OUT, ARTIFACTS
    specs = [{**study.condition('gelu',128,seed,dataset='new2k',layers=0),'activation':'gelu'} for seed in plan['seeds']]
    write_json(OUT / 'status.json',dict(state='running',new_runs=3,reused_controls=3,started_unix=time.time()))
    def lane(gpu):
        for spec in specs[gpu::2]:
            execute(spec,plan,gpu)
        write_json(OUT / f'gpu{gpu}.json',dict(state='complete',updated_unix=time.time()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane,gpu) for gpu in range(2)]
        for future in futures:
            future.result()
    groups = {}
    for activation, conditions in [('none',plan['controls']),('gelu',specs)]:
        values = [study.measure(spec,plan) for spec in conditions]
        groups[activation] = dict(per_seed=values, **{f'{metric}_{stat}':float(fn([v[metric] for v in values]))
            for metric in ['ap','iou'] for stat,fn in [('mean',np.mean),('sd',lambda x:np.std(x,ddof=1))]})
    deltas = [dict(seed=a['seed'], **{k:float(b[k]-a[k]) for k in ['ap','iou']}) for a,b in
        zip(groups['none']['per_seed'],groups['gelu']['per_seed'])]
    verify(plan)
    write_json(OUT / 'summary.json',dict(groups=groups,paired_gelu_minus_linear=deltas,
        manifest_sha256=digest(OUT / 'manifest.json'), uncertainty='sample SD over seeds, not dataset confidence intervals'))
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
            print('Frozen: three GELU runs, three completed controls; 2,000 s train + 800 s validation.')
        else:
            run(plan)
    except Exception as error:
        if not child:
            write_json(OUT / 'status.json',dict(state='failed',error=str(error),updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
