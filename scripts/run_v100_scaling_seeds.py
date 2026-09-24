#!/usr/bin/env python3
"""Missing scaling seeds on packed, verified copies of seed-0 inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import run_scaling_seeds as study
from run_v100_lr_sweep import frozen, read, sha, write

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/george/DATA/songmae-lr-sweep-20260916')
study.SOURCE = SOURCE


class PackedWindows:
    def __init__(self, rows, shards, config):
        ids = [r['recording'] for r in rows]
        for partition in ['train', 'validation']:
            folder = SOURCE/'data'/partition
            positions = {name:i for i,name in enumerate(read(folder/'index.json')['recordings'])}
            if set(ids) <= positions.keys():
                self.indices = [positions[name] for name in ids]
                self.specs = np.load(folder/'spec.npy', mmap_mode='r')
                self.targets = np.load(folder/'target.npy', mmap_mode='r')
                break
        else:
            raise ValueError('Unknown or mixed packed partition')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return (torch.from_numpy(self.specs[i].copy()).unsqueeze(0),
                torch.from_numpy(self.targets[i].astype(np.float32)), 1000)


def install_inputs():
    import birdsong_detect_distill.data as data
    original = data.read_rows
    def fixed_rows(path, seed=0, maximum=None):
        partition = 'validation' if Path(path).name == 'validation.jsonl' else 'train'
        return original(SOURCE/'data'/partition/'labels.jsonl', 0, maximum)
    data.read_rows = fixed_rows
    data.PixelWindows = PackedWindows


def prepare():
    from birdsong_detect_distill.data import read_rows
    from birdsong_detect_distill.full_encoder_linear import encoder_state
    from birdsong_detect_distill.model import load_backbone
    from birdsong_detect_distill.pointwise_bare_linear import BareLinearHead
    from run_last_block_linear_2k import state_hash
    destination = study.OUT/'manifest.json'
    if destination.exists():
        plan = read(destination)
        for p, expected in plan['protected_files'].items():
            if sha(p) != expected:
                raise ValueError('Frozen input changed: '+p)
        return plan
    plan = read(ROOT/'seed0_manifest.json')
    paper = read(SOURCE/'source_manifest.json')
    protected = {str(ROOT/'seed0_manifest.json'):sha(ROOT/'seed0_manifest.json')}
    for partition, key in [('train','25000'),('validation','validation')]:
        print('Verifying packed '+partition+' tensors',flush=True)
        folder = SOURCE/'data'/partition
        rows = read_rows(folder/('validation.jsonl' if partition == 'validation' else '25000.jsonl'))
        ids = [r['recording'] for r in rows]
        if partition == 'train':
            for budget, item in plan['datasets']['budgets'].items():
                if ids[:int(budget)//5] != item['recordings']:
                    raise ValueError('Seed-0 training selection changed')
        elif sorted(ids) != plan['datasets']['validation']['recording_ids']:
            raise ValueError('Validation selection changed')
        data = PackedWindows(rows, None, None)
        a, b = hashlib.sha256(), hashlib.sha256()
        for spec, target, valid in data:
            a.update(spec.numpy().tobytes())
            b.update(target.numpy().tobytes())
        expected = paper['datasets'][key]
        if (a.hexdigest(), b.hexdigest()) != (expected['preflight_input_sha256'], expected['preflight_mask_sha256']):
            raise ValueError('Packed tensors differ from original training')
        for name in ['index.json','labels.jsonl','spec.npy','target.npy']:
            p = folder/name
            protected[str(p)] = sha(p)
    initializations = {}
    for size, model in plan['models'].items():
        for seed in [1,2]:
            print(f'Checking {size} seed {seed} initialization',flush=True)
            torch.manual_seed(seed)
            backbone = load_backbone(model['backbone'],torch.device('cpu'),model['revision'])
            head = BareLinearHead(backbone.config.enc_hidden_d,0,4,1000,32,1,dropout=.1,layers=0)
            enc_sha = state_hash(encoder_state(backbone))
            if enc_sha != plan['initializations'][size]['initial_encoder_sha256']:
                raise ValueError('Pretrained encoder differs')
            initializations[f'{size}_s{seed}'] = dict(initial_head_sha256=state_hash(head.state_dict()),
                initial_encoder_sha256=enc_sha,encoder_depth=len(backbone.songmae.encoder.layers))
            del backbone, head
    plan.update(run='v100_scaling_seeds12_20260919',seeds=[1,2],jobs=study.JOBS,
        initializations=initializations, protected_files=protected,
        storage='Best checkpoint and epoch-boundary resume only; no redundant epoch trajectories.',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ['scripts','src'] for p in (ROOT/folder).rglob('*.py')})
    frozen(destination,plan)
    print('Manifest verified and frozen',flush=True)
    return plan


def lane(gpu):
    time.sleep(gpu*20)
    # One size per GPU, with independent worker processes and epoch-boundary resumes.
    for job, recipe in study.JOBS.items():
        if recipe['size'] != study.SIZES[gpu]:
            continue
        status = study.OUT/'runs'/job/'status.json'
        if status.exists() and read(status)['state'] == 'complete':
            continue
        with (study.OUT/'logs'/f'{job}.log').open('a') as log:
            result = subprocess.run([sys.executable,'-u',str(Path(__file__)),'--job',job],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{job} failed; inspect its log before retrying')


if __name__ == '__main__':
    if os.uname().nodename != 'george-server' or ROOT != Path('/media/george/DATA/songmae-scaling-seeds12-20260919'):
        raise SystemExit('Dedicated V100 directory only')
    os.chdir(ROOT)
    install_inputs()
    study.prepare = prepare
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
    parser.add_argument('--lane',type=int,choices=[0,1,2])
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    study.OUT.mkdir(exist_ok=True)
    (study.OUT/'logs').mkdir(exist_ok=True)
    if args.prepare:
        prepare()
    elif args.job:
        study.worker(args.job)
    elif args.lane is not None:
        lane(args.lane)
    else:
        lock = (study.OUT/'driver.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(study.OUT/'status.json',dict(state='preparing',updated_unix=time.time()))
            prepare()
            write(study.OUT/'status.json',dict(state='running',updated_unix=time.time()))
            with ThreadPoolExecutor(3) as pool:
                futures = [pool.submit(lane,gpu) for gpu in range(3)]
                for future in futures:
                    future.result()
            rows = [read(study.OUT/'runs'/job/'result.json') for job in study.JOBS]
            write(study.OUT/'scaling.json',rows)
            write(study.OUT/'status.json',dict(state='complete',runs=len(rows),updated_unix=time.time()))
        except BaseException as error:
            write(study.OUT/'status.json',dict(state='failed',error=repr(error),updated_unix=time.time()))
            raise
