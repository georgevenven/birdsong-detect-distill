#!/usr/bin/env python3
"""Detached V100-only staged LR sweep, with matched-baseline three-seed confirmation."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
BASELINE = (1e-5, 1e-3)
ENCODERS = [1e-5, 3e-6, 3e-5]
HEADS = [3e-4, 3e-3]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def frozen(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('Frozen configuration changed: ' + str(path))
    write(path, value)


def name(size, encoder, head, seed):
    return f'{size}_enc{encoder:g}_head{head:g}_s{seed}'


def worker(size, encoder, head_lr, seed):
    import numpy as np
    import torch
    import run_full_encoder_backbones_snapshot as trainer
    from torch.utils.data import Dataset, DataLoader
    from birdsong_detect_distill.full_encoder_linear import load_heads
    from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate

    job = name(size, encoder, head_lr, seed)
    out = OUT / 'runs' / job; artifacts = ROOT / 'artifacts' / job
    out.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    lock = (out / 'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    master = read(ROOT / 'source_manifest.json')
    plan = dict(run='v100_lr_sweep_2026-09-16', seed=seed, epochs=10, models=master['models'],
        datasets=dict(train=master['datasets']['25000'], validation=master['datasets']['validation']),
        training_config={**master['training_config'], 'backbone_learning_rate':encoder, 'learning_rate':head_lr},
        bundle_sha256=sha(ROOT / 'bundle.json'), policy_sha256=sha(OUT / 'policy.json'),
        execution='FP32 training, same effective batch 16/microbatch 2; memory-mapped exact tensors, zero loader workers.',
        software={key:version(key) for key in ['torch','transformers','numpy','scipy','librosa']})
    frozen(out / 'manifest.json', plan)

    class Packed(Dataset):
        def __init__(self, partition):
            folder = ROOT / 'data' / partition
            self.spec = np.load(folder / 'spec.npy', mmap_mode='r')
            self.target = np.load(folder / 'target.npy', mmap_mode='r')
            expected = plan['datasets'][partition]
            index = read(folder / 'index.json')
            if (len(self.spec) * 5 != expected['seconds'] or sorted(index['recordings']) != expected['recording_ids']):
                raise ValueError('Packed split differs from original')
        def __len__(self):
            return len(self.spec)
        def __getitem__(self, index):
            return torch.from_numpy(self.spec[index].copy())[None], torch.from_numpy(self.target[index].astype(np.float32)), 1000

    def dataset(plan, partition, config):
        expected = read(ROOT / 'hf/hub' / ('models--' + plan['models']['large']['backbone'].replace('/', '--')) /
            'snapshots' / plan['models']['large']['revision'] / 'config.json')
        if any(getattr(config, k) != expected[k] for k in ['audio_mean','audio_std','mels','num_timebins','patch_height','patch_width']):
            raise ValueError('Model normalization/grid differs')
        return Packed(partition)

    def loader(*args, **kwargs):
        return DataLoader(*args, **{**kwargs, 'num_workers':0})

    trainer.OUT = out
    trainer.original.previous.dataset = dataset
    trainer.DataLoader = loader
    checkpoint = artifacts / 'model.pt'
    if not checkpoint.exists() and torch.cuda.mem_get_info(0)[0] < 13 * 1024**3:
        raise ValueError('V100 GPU occupied; no other job will be stopped')
    trainer.train(plan, size, out, artifacts, checkpoint)
    torch.cuda.empty_cache()
    if not (out / 'loro.json').exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        anchor = read(ROOT / 'anchor.json'); reference = anchor['protocol']
        protocol = dict(manifest=plan, checkpoint={k:v for k,v in saved.items() if k not in ['encoder','head']},
            checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint), anchor=str(ROOT / 'anchor.json'),
            anchor_sha256=sha(ROOT / 'anchor.json'), student_inference=read(ROOT / 'inference.json'))
        frozen(out / 'protocol.json', protocol)
        write(out / 'status.json', dict(state='evaluating', job=job, updated_unix=time.time()))
        trainer.scorer.load_heads = load_heads
        trainer.scorer.score_checkpoint(checkpoint, saved, out, artifacts / 'predictions', ROOT / 'raw', reference, anchor, 'samples')
        report = read(out / 'comparison.json')
        spec = dict(id=job, family='full_encoder_linear', label=size, seed=seed, threshold_floor_index=0, path=out/'comparison.json')
        write(out / 'loro.json', cross_calibrate(spec, report, checked_rows(report)))
    write(out / 'status.json', dict(state='complete', job=job, updated_unix=time.time()))


def result(size, encoder, head, seed):
    out = OUT / 'runs' / name(size, encoder, head, seed)
    score = read(out / 'loro.json')['summary']
    checkpoint = read(out / 'checkpoint.json')
    return dict(job=out.name, size=size, encoder_lr=encoder, head_lr=head, seed=seed,
        selected_epoch=checkpoint['metrics']['selected_epoch'], validation_bce=checkpoint['metrics']['best_val_loss'],
        checkpoint_sha256=checkpoint['sha256'], **{k:score[k] for k in ['ap','iou']})


def invoke(size, encoder, head, seed, gpu):
    job = name(size, encoder, head, seed)
    if (OUT / 'runs' / job / 'status.json').exists() and read(OUT / 'runs' / job / 'status.json')['state'] == 'complete':
        return result(size, encoder, head, seed)
    while True:
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
        if available > 2 * 1024**2:
            break
        write(OUT / f'{size}.json', dict(state='waiting_for_host_memory', available_kib=available))
        time.sleep(30)
    write(OUT / f'{size}.json', dict(state='running', job=job, gpu=gpu, updated_unix=time.time()))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu)}
    with (OUT / 'logs' / f'{job}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(Path(__file__)), '--worker', size,
            '--encoder-lr', str(encoder), '--head-lr', str(head), '--seed', str(seed)],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return result(size, encoder, head, seed)


def lane(size, gpu):
    trials = [invoke(size, lr, BASELINE[1], 0, gpu) for lr in ENCODERS]
    winner = max(trials, key=lambda row:row['ap'])
    frozen(OUT / f'{size}_encoder_selection.json', dict(trials=trials, selected=winner))
    trials += [invoke(size, winner['encoder_lr'], lr, 0, gpu) for lr in HEADS]
    winner = max(trials, key=lambda row:row['ap'])
    frozen(OUT / f'{size}_selection.json', dict(trials=trials, selected=winner,
        rule='Seed-0 full Powdermill pixel AP; freeze before confirmation seeds. Not an external-test result.'))
    configurations = list(dict.fromkeys([BASELINE, (winner['encoder_lr'], winner['head_lr'])]))
    confirmed = []
    for encoder, head in configurations:
        rows = [invoke(size, encoder, head, seed, gpu) for seed in [0,1,2]]
        confirmed.append(dict(encoder_lr=encoder, head_lr=head, rows=rows,
            **{k:dict(mean=mean(r[k] for r in rows), sd=stdev(r[k] for r in rows)) for k in ['ap','iou']}))
    write(OUT / f'{size}_complete.json', dict(size=size, trials=trials, confirmed=confirmed))
    write(OUT / f'{size}.json', dict(state='complete', updated_unix=time.time()))


def driver():
    if platform.node() != 'george-server':
        raise ValueError('This driver may run only on the V100 host, never Twins')
    OUT.mkdir(exist_ok=True); (OUT / 'logs').mkdir(exist_ok=True)
    lock = (OUT / 'driver.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bundle = read(ROOT / 'bundle.json')
    for relative, expected in bundle['files'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Transfer/source checksum mismatch: ' + relative)
    frozen(OUT / 'policy.json', dict(bundle_sha256=sha(ROOT / 'bundle.json'), encoders=ENCODERS,
        heads=HEADS, baseline=list(BASELINE), sizes=['large','base','micro'], training_seconds=25000,
        validation_seconds=2500, epochs=10, checkpoint='Minimum XC validation BCE',
        selection='Staged encoder then head search using seed 0 Powdermill pixel AP; baseline and selected pair confirmed with seeds 1/2.',
        caveat='Development-set hyperparameter tuning, not nested CV, not exhaustive joint LR optimization. Same seed used to select and included in three-seed report; report confirmation-only seeds separately.',
        external='No external test datasets used or evaluated; Twins suite unchanged.',
        qwen='V100 idle server gracefully stopped; no automatic Qwen restart.'))
    write(OUT / 'status.json', dict(state='running', updated_unix=time.time()))
    try:
        with ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(lane, size, gpu) for gpu,size in enumerate(['large','base','micro'])]
            for future in futures:
                future.result()
        reports = [read(OUT / f'{size}_complete.json') for size in ['large','base','micro']]
        write(OUT / 'summary.json', dict(models=reports, completed_unix=time.time()))
        lines = ['# V100 learning-rate sweep', '',
            '25,000 s train; 2,500 s XC validation; 10 epochs, best XC BCE. Full-Powdermill AP/IoU.',
            'Seed 0 selects settings; seeds 1/2 are confirmation. No external-test selection.', '',
            '| Size | Encoder LR | Head LR | Pixel AP (3 seeds) | IoU (3 seeds) | AP (confirmation seeds 1/2) |',
            '|---|---:|---:|---:|---:|---:|']
        for report in reports:
            for row in report['confirmed']:
                lines.append(f"| {report['size']} | {row['encoder_lr']:g} | {row['head_lr']:g} | "
                    f"{row['ap']['mean']:.4f} ± {row['ap']['sd']:.4f} | {row['iou']['mean']:.4f} ± {row['iou']['sd']:.4f} | "
                    f"{mean(r['ap'] for r in row['rows'] if r['seed'] != 0):.4f} |")
        (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
        write(OUT / 'status.json', dict(state='complete', updated_unix=time.time()))
    except BaseException as error:
        write(OUT / 'status.json', dict(state='failed', error=repr(error), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['micro','base','large'])
    parser.add_argument('--encoder-lr', type=float)
    parser.add_argument('--head-lr', type=float)
    parser.add_argument('--seed', type=int, choices=[0,1,2])
    args = parser.parse_args(); os.chdir(ROOT)
    if args.worker:
        worker(args.worker, args.encoder_lr, args.head_lr, args.seed)
    else:
        driver()
