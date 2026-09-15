#!/usr/bin/env python3
"""Remove the detector LayerNorm only; same linear initialization, data and Powdermill protocol."""
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
from birdsong_detect_distill.pointwise_bare_linear import ARCHITECTURE, BareLinearHead, load_heads
from prepare_detector_study import verify
from run_pointwise_single_linear_2k import OUT as PARENT, ARTIFACTS as PARENT_ARTIFACTS

RUN = 'pointwise_bare_linear_2000train_800val_2026-09-14'
OUT, ARTIFACTS = PARENT.parent / RUN, PARENT_ARTIFACTS.parent / RUN
SCRIPT = 'scripts/run_pointwise_bare_linear_2k.py'
MODEL = 'src/birdsong_detect_distill/pointwise_bare_linear.py'


def prepare():
    destination = OUT / 'manifest.json'
    if destination.exists():
        plan = json.loads(destination.read_text())
        verify(plan)
        return plan
    parent = json.loads((PARENT / 'manifest.json').read_text())
    verify(parent)
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('LayerNorm plus linear controls must be complete')
    protected, controls = dict(parent['protected_files']), []
    for seed in parent['seeds']:
        path = PARENT / 'conditions' / f'single_linear_d0_new2k_s{seed}.json'
        spec = json.loads(path.read_text())
        shared.study.measure(spec,parent)
        controls.append({**spec,'reused':True})
        for p in [path,Path(spec['checkpoint']),Path(spec['report'])]:
            protected[str(p)] = digest(p)
    for p in [PARENT / 'manifest.json',PARENT / 'status.json',PARENT / 'summary.json']:
        protected[str(p)] = digest(p)
    plan = {**parent,'run':RUN,'controls':controls,'artifacts':str(ARTIFACTS),'protected_files':protected,
        'code_sha256':{**parent['code_sha256'],SCRIPT:digest(SCRIPT),MODEL:digest(MODEL)},
        'architecture':'Linear(768,32) with bias on raw final SongMAE tokens; no detector normalization',
        'ablation':'Remove detector LayerNorm only; no hidden projection, positions, GELU, attention or dropout',
        'initialization':'Same initial linear weights/bias as LayerNorm+Linear controls; no trained weights reused',
        'pairing':'Same retained parameter values, sample order and CPU RNG; initial functions differ without normalization',
        'backbone_normalization':'Pre-LN within each block; default final residual output has no terminal LayerNorm. Optional normalized average_top_k is not used.',
        'head_parameters':24608,'control_head_parameters':26144}
    verify(plan)
    write_json(destination,plan)
    return plan


def check_pair(saved, control, audit):
    old = torch.load(control['checkpoint'],map_location='cpu',weights_only=True)
    if audit['control_initial_sha256'] != old['initial_head_sha256'] or audit['bare_initial_sha256'] != saved['initial_head_sha256']:
        raise ValueError('linear initialization does not match the LayerNorm control')
    if set(saved['head']) != {'output.weight','output.bias'} or sum(v.numel() for v in saved['head'].values()) != 24608:
        raise ValueError('unexpected bare-linear architecture')
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
            head = BareLinearHead(*args,**kwargs)
            shared.study.frozen_json(audit_path,dict(control_initial_sha256=head.control_initial_sha256,
                bare_initial_sha256=hashlib.sha256(b''.join(
                    v.detach().cpu().numpy().tobytes() for v in head.state_dict().values())).hexdigest()))
            return head
        shared.trainer.DenseHead = factory
        sys.argv = ['scripts/train.py','--annotations',plan['datasets']['new2k']['path'],
            '--validation-annotations',plan['datasets']['validation']['path'],'--backbone',plan['backbone'],
            '--backbone-revision',plan['revision'],'--hidden','0','--head-layers','0',
            '--seed',str(spec['seed']),'--epochs','5','--learning-rate','0.001','--weight-decay','0.0001',
            '--batch-size','16','--eval-batch-size','4','--accumulation','1','--dropout','0.1',
            '--tv-weight','0','--no-target-smoothing','--log-every','5','--out',str(pending)]
        shared.trainer.main()
    saved = torch.load(pending,map_location='cpu',weights_only=True)
    audit = json.loads(audit_path.read_text())
    control = next(c for c in plan['controls'] if c['seed']==spec['seed'])
    check_pair(saved,control,audit)
    BareLinearHead(768,0,4,1000,32,1,layers=0).load_state_dict(saved['head'])
    saved.update(detector_architecture=ARCHITECTURE,head_activation='none',detector_position_embeddings=False,
        layer_norm=False,hidden_projection=None,initialization_audit=audit,
        experiment_code_sha256=plan['code_sha256'],paired_control_sha256=digest(control['checkpoint']))
    staged = target.with_suffix('.tagged.pt')
    torch.save(saved,staged)
    staged.replace(target)
    print(f'Validated and tagged bare-linear checkpoint: {target}',flush=True)


def evaluate(path, spec, plan):
    saved = torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
    if saved.get('detector_architecture') != ARCHITECTURE or saved.get('layer_norm') is not False or saved.get('experiment_code_sha256') != plan['code_sha256']:
        raise ValueError('incorrect architecture or experiment provenance')
    check_pair(saved,next(c for c in plan['controls'] if c['seed']==spec['seed']),saved['initialization_audit'])
    shared.evaluator.OUT, shared.scorer.load_heads = OUT, load_heads
    shared.evaluator.evaluate(path)
    shared.study.measure(spec,plan)


def run(plan):
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
    specs = [shared.study.condition('bare_linear',0,seed,dataset='new2k',layers=0) for seed in plan['seeds']]
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
    for name,conditions in [('layernorm_linear',plan['controls']),('bare_linear',specs)]:
        values = [shared.study.measure(spec,plan) for spec in conditions]
        groups[name] = dict(per_seed=values,**{f'{metric}_{stat}':float(fn([v[metric] for v in values]))
            for metric in ['ap','iou'] for stat,fn in [('mean',np.mean),('sd',lambda x:np.std(x,ddof=1))]})
    deltas = [dict(seed=a['seed'],**{k:float(b[k]-a[k]) for k in ['ap','iou']}) for a,b in
        zip(groups['layernorm_linear']['per_seed'],groups['bare_linear']['per_seed'])]
    verify(plan)
    write_json(OUT / 'summary.json',dict(groups=groups,paired_bare_minus_layernorm=deltas,
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
            print('Frozen: three bare-linear runs, 2,000 s train + 800 s validation, LayerNorm+Linear controls.')
        else:
            run(plan)
    except Exception as error:
        if not child:
            write_json(OUT / 'status.json',dict(state='failed',error=str(error),updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
