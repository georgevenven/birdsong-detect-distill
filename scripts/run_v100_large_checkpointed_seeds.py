#!/usr/bin/env python3
"""Memory-only recovery for Large seeds; keep the original BF16/batch recipe."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint

import run_v100_scaling_seeds as driver
from run_v100_lr_sweep import read, write
import birdsong_detect_distill.model as model

study = driver.study
original_loader = model.load_backbone


def load_backbone(*args, **kwargs):
    backbone = original_loader(*args, **kwargs)
    for layer in backbone.songmae.encoder.layers:
        forward = layer.forward_features
        def recompute(*values, _forward=forward, **keywords):
            if torch.is_grad_enabled():
                return checkpoint(_forward, *values, use_reentrant=False, preserve_rng_state=True, **keywords)
            return _forward(*values, **keywords)
        layer.forward_features = recompute
    return backbone


if __name__ == '__main__':
    if os.uname().nodename != 'george-server' or driver.ROOT != Path('/media/george/DATA/songmae-scaling-large-seeds12-20260919'):
        raise SystemExit('Dedicated Large recovery directory only')
    os.chdir(driver.ROOT)
    driver.install_inputs()
    model.load_backbone = load_backbone
    study.JOBS = {job:cfg for job,cfg in study.JOBS.items() if cfg['size']=='large'}
    study.epoch_save = lambda *args, **kwargs: None
    original_eval = study.eval_best
    def evaluate(plan, out, artifacts):
        epoch = read(out/'checkpoint.json')['metrics']['selected_epoch']
        alias = artifacts/f'epoch_{epoch:02d}.pt'
        if not alias.exists():
            alias.symlink_to('model.pt')
        original_eval(plan,out,artifacts)
    study.eval_best = evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',choices=study.JOBS)
    args = parser.parse_args()
    study.OUT.mkdir(exist_ok=True)
    (study.OUT/'logs').mkdir(exist_ok=True)
    if args.job:
        study.worker(args.job)
    else:
        lock = (study.OUT/'driver.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(study.OUT/'status.json',dict(state='preparing',updated_unix=time.time()))
            driver.prepare()
            write(study.OUT/'memory_runtime.json',dict(activation_checkpointing='Every encoder block; non-reentrant; RNG preserved',
                reason='Original V100 Large BF16 attention exceeded 16GB before first epoch',
                batch_size=16,microbatch_size=2,precision='BF16 autocast; FP32 master weights and validation'))
            for job in study.JOBS:
                status = study.OUT/'runs'/job/'status.json'
                if status.exists() and read(status)['state']=='complete':
                    continue
                write(study.OUT/'status.json',dict(state='running',job=job,updated_unix=time.time()))
                with (study.OUT/'logs'/f'{job}.log').open('a') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--job',job],check=True,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':'2'},stdout=log,stderr=subprocess.STDOUT)
            write(study.OUT/'scaling.json',[read(study.OUT/'runs'/job/'result.json') for job in study.JOBS])
            write(study.OUT/'status.json',dict(state='complete',runs=len(study.JOBS),updated_unix=time.time()))
        except BaseException as error:
            write(study.OUT/'status.json',dict(state='failed',error=repr(error),updated_unix=time.time()))
            raise
