#!/usr/bin/env python3
"""Matched zero/one-layer heads on a frozen snapshot of the new XC teacher labels."""
import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoConfig

import evaluate_detector_study as evaluator
import qwen_prompt_study as teacher
import run_detector_study as study
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.data import PixelWindows, load_spec_slice, read_rows
from prepare_detector_study import CODE, OUT as PARENT, verify
from run_qwen_xc_50k import validate
from summarize_three_seed_figures import unpack

RUN = 'new_teacher_2000train_800val_2026-09-14'
OUT = Path('results/qwen_teacher_powdermill') / RUN
LABELS = Path('data/annotations/xcl') / RUN
# Keep prediction arrays off the nearly full root filesystem.
ARTIFACTS = Path('/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments') / RUN
QUEUE = Path('data/annotations/xcl/powdermill_protocol_50000s_2026-09-14')
SCRIPT = 'scripts/run_new_teacher_2k.py'


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        verify(plan)
        return plan
    parent = json.loads((PARENT / 'manifest.json').read_text())
    queue = json.loads((QUEUE / 'manifest.json').read_text())
    queue_hash = digest(QUEUE / 'manifest.json')
    selection = json.loads((QUEUE / 'selection.json').read_text())
    excluded = set().union(*(set(v.get('xc_ids', [])) for v in selection['exclusions'].values()))
    snapshot_path = OUT / 'snapshot.json'
    if not snapshot_path.exists():
        completed = {p.stem for p in (QUEUE / 'annotations/self_review_1').glob('*.json')}
        names = [w['name'] for w in queue['windows'] if w['name'] in completed]
        random.Random(17).shuffle(names)
        write_json(snapshot_path,dict(names=names,created_unix=time.time(),queue_sha256=queue_hash))
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot['queue_sha256'] != queue_hash:
        raise ValueError('queue changed after snapshot')
    by_name = {w['name']:w for w in queue['windows']}
    pool = [by_name[name] for name in snapshot['names']]
    if len(pool) < 560:
        raise ValueError('need 400 completed training windows and 160 separate validation windows')
    protected = {str(p):digest(p) for p in [QUEUE / 'manifest.json', QUEUE / 'selection.json',
        snapshot_path, PARENT / 'manifest.json', Path(parent['anchor']), Path(parent['reuse']['w384_s0']['report'])]}
    config = AutoConfig.from_pretrained(parent['backbone'], revision=parent['revision'],
        trust_remote_code=True, local_files_only=True)
    datasets = {}
    for name, windows in [('new2k', pool[:400]), ('validation', pool[400:560])]:
        rows = []
        for window in windows:
            for stage in ['reasoning', 'self_review_1']:
                if not validate(QUEUE, window, queue_hash, stage):
                    raise ValueError('missing accepted annotation: ' + window['name'])
                source = QUEUE / 'annotations' / stage / f'{window["name"]}.json'
                protected[str(source)] = digest(source)
            annotation = json.loads(source.read_text())
            if annotation['events'] != teacher.parsed_events(annotation['raw_annotation'], window):
                raise ValueError('mapped events differ from canonical raw teacher boxes')
            recording, shard, start, end, left, right = window['tile']
            if recording in excluded or right-left != 1000 or not 0 <= left < right <= end-start:
                raise ValueError('excluded recording or invalid five-second window')
            raw = load_spec_slice(Path(queue['spec_dir']) / 'shards' / shard, start+left, start+right)
            if hashlib.sha256(raw.tobytes(order='C')).hexdigest() != window['spectrogram_sha256']:
                raise ValueError('teacher/student input spectrogram mismatch')
            rows.append(dict(status='ok', recording=recording, source=dict(shard=shard,start=start,end=end),
                tile=dict(start_timebin=left,end_timebin=right,ownership_start_timebin=left,ownership_end_timebin=right),
                events=annotation['events'], annotation=str(source), annotation_sha256=digest(source),
                manifest_sha256=queue_hash, worker_role=annotation['worker_role'],
                worker_backend_sha256=annotation['worker_backend_sha256']))
        destination = LABELS / f'{name}.jsonl'
        content = ''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_text() != content:
            raise ValueError('partial frozen split exists; do not overwrite it')
        if not destination.exists():
            destination.write_text(content)
        data = PixelWindows(read_rows(destination), 'data/xcl/shards', config)
        ids = sorted({r['recording'] for r in rows})
        if len(data) != len(windows) or len(ids) != len(windows) or sum(w[2] for w in data.windows) != len(windows)*1000:
            raise ValueError('loader changed the exact budget or recordings are duplicated')
        inputs, masks = hashlib.sha256(), hashlib.sha256()
        for i in range(len(data)):
            audio, target, valid = data[i]
            if not torch.isfinite(audio).all() or not torch.all((target == 0) | (target == 1)):
                raise ValueError('invalid student input or nonbinary target')
            inputs.update(audio.numpy().tobytes())
            masks.update(target[:, :valid].numpy().tobytes())
        datasets[name] = dict(path=str(destination), sha256=digest(destination), seconds=len(windows)*5,
            windows=len(data), recording_ids=ids, supervision=data.supervision_counts(),
            preflight_input_sha256=inputs.hexdigest(), preflight_mask_sha256=masks.hexdigest(),
            focal_species=len({w['metadata']['ebird_code'] for w in windows}),
            coordinate_sites=len({w['coordinate_site_0_01_degree'] for w in windows}),
            windows_selected=[w['name'] for w in windows])
        protected[str(destination)] = digest(destination)
        del data
    if set(datasets['new2k']['recording_ids']) & set(datasets['validation']['recording_ids']):
        raise ValueError('training/validation recording overlap')
    inherited = ['backbone','revision','seeds','epochs','learning_rate','training_config',
        'anchor','checkpoint_selection','evaluation','inference','caveat','reuse']
    plan = {k:parent[k] for k in inherited}
    plan.update(run=RUN, widths=[128], layers=[1,0], datasets=datasets, protected_files=protected,
        code_sha256={p:digest(p) for p in [*CODE, SCRIPT, 'scripts/qwen_prompt_study.py', 'scripts/run_qwen_xc_50k.py']},
        selection=dict(seed=17, snapshot_unix=snapshot['created_unix'], completed_pool_windows=len(pool),
            completed_pool_names=[w['name'] for w in pool],
            rule='shuffle completed windows in manifest order; first 400 train, next 160 validation',
            caveat='preliminary completed-pool sample; annotation latency can affect availability'),
        teacher='new Powdermill-protocol XC reasoning plus one numbered self-review; hard foreground boxes',
        architecture='existing pointwise zero-layer head versus one self-attention layer; neither is cross-attention',
        pairing='same seed, input split, shared-weight initialization and training order; vary attention layer only',
        gpu_policy='Twins only; leave V100 annotations running; do not automatically restart Twins Qwen',
        artifacts=str(ARTIFACTS))
    verify(plan)
    write_json(path, plan)
    return plan


def execute(spec, plan, gpu):
    path = OUT / 'conditions' / f'{spec["name"]}.json'
    study.frozen_json(path, spec)
    if shutil.disk_usage(ARTIFACTS).free < 2*1024**3 or shutil.disk_usage('.').free < 1024**3:
        raise ValueError('low disk space; stop without deleting existing work')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'HF_HUB_OFFLINE':'1',
        'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'PYTHONUNBUFFERED':'1'}
    commands = []
    if not Path(spec['checkpoint']).exists():
        commands.append(('train', ['scripts/train.py', '--annotations',plan['datasets']['new2k']['path'],
            '--validation-annotations',plan['datasets']['validation']['path'], '--backbone',plan['backbone'],
            '--backbone-revision',plan['revision'], '--hidden','128', '--head-layers',str(spec['layers']),
            '--seed',str(spec['seed']), '--epochs','5', '--learning-rate','0.001', '--weight-decay','0.0001',
            '--batch-size','16', '--eval-batch-size','4', '--accumulation','1', '--dropout','0.1',
            '--tv-weight','0', '--no-target-smoothing', '--log-every','5', '--out',spec['checkpoint']]))
    commands.append(('evaluate', [SCRIPT, '--evaluate',str(path)]))
    for action, args in commands:
        verify(plan)
        write_json(OUT / f'gpu{gpu}.json',dict(state=action, condition=spec['name'], updated_unix=time.time()))
        print(f'{action}: {spec["name"]} on Twins GPU {gpu}',flush=True)
        with (OUT / 'logs' / f'{action}_{spec["name"]}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    study.measure(spec, plan)


def run(plan):
    for unit in ['twins','server']:
        state = subprocess.check_output(['systemctl','--user','show',
            f'birdsong-qwen-xc50k-{unit}-20260914.service','-p','ActiveState','--value'],text=True).strip()
        if state != 'inactive':
            raise ValueError('Twins Qwen must finish graceful shutdown before GPU work')
    for gpu in range(2):
        if torch.cuda.mem_get_info(gpu)[0] < 20*1024**3:
            raise ValueError('Twins GPU is occupied; do not interfere')
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    study.OUT, study.ARTIFACTS = OUT, ARTIFACTS
    specs = [study.condition(f'layers{layer}',128,seed,dataset='new2k',layers=layer)
        for seed in plan['seeds'] for layer in plan['layers']]
    write_json(OUT / 'status.json',dict(state='running',conditions=[s['name'] for s in specs],started_unix=time.time()))
    def lane(gpu):
        for spec in specs[gpu::2]:
            execute(spec,plan,gpu)
        write_json(OUT / f'gpu{gpu}.json',dict(state='complete',updated_unix=time.time()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane,gpu) for gpu in range(2)]
        for future in futures:
            future.result()
    for seed in plan['seeds']:
        pair = [unpack(Path(s['report']))[0] for s in specs if s['seed']==seed]
        for a,b in zip(pair[0]['training_history'],pair[1]['training_history']):
            if any(a[k]!=b[k] for k in ['epoch','sample_order_sha256','cpu_rng_sha256']):
                raise ValueError('paired training sample order or CPU randomness differs')
    study.summarize_stage('pointwise',sorted(specs,key=lambda s:(-s['layers'],s['seed'])),plan,'layers')
    verify(plan)
    write_json(OUT / 'status.json',dict(state='complete',runs=6,training_seconds=2000,completed_unix=time.time()))
    print('Complete. Twins Qwen remains paused; V100 was not changed.',flush=True)


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.evaluate:
        lock = (OUT / 'driver.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = prepare()
        if args.prepare:
            print(json.dumps({k:{f:v[f] for f in ['seconds','windows','focal_species','coordinate_sites']}
                for k,v in plan['datasets'].items()},indent=2))
        elif args.evaluate:
            evaluator.OUT = OUT
            evaluator.evaluate(args.evaluate)
        else:
            run(plan)
    except Exception as error:
        if not args.evaluate:
            write_json(OUT / 'status.json',dict(state='failed',error=str(error),updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
