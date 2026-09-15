#!/usr/bin/env python3
"""Compare zero/one-layer heads at 1,000 s without changing the completed 10k study."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoConfig

import evaluate_detector_study as evaluator
import run_detector_study as study
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.data import PixelWindows, read_rows
from prepare_detector_study import OUT as PARENT, verify
from summarize_three_seed_figures import unpack

RUN = 'pointwise_1000train_800val_2026-09-11'
OUT = Path('results/qwen_teacher_powdermill') / RUN
ARTIFACTS = Path('artifacts') / RUN
PAUSE = Path('/home/george-vengrovski/.config/birdsong-detect-distill/qwen.paused')
SCRIPT = 'scripts/run_pointwise_1k.py'


def prepare():
    manifest = OUT / 'manifest.json'
    if manifest.exists():
        plan = json.loads(manifest.read_text())
        verify(plan)
        return plan
    parent = json.loads((PARENT / 'manifest.json').read_text())
    verify(parent)
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('the parent study must be complete')
    scaling_path = Path('data/annotations/xcl/scaling_10000train_800val_2026-09-10/manifest.json')
    scaling = json.loads(scaling_path.read_text())
    budget = scaling['budgets']['1000']
    source, validation = budget['file'], parent['datasets']['validation']
    if (scaling['source_annotations']['sha256'] != parent['datasets']['original10k']['sha256']
            or scaling['validation_annotations']['sha256'] != validation['sha256']
            or digest(source['path']) != source['sha256']):
        raise ValueError('changed scaling subset or parent split')
    rows = read_rows(source['path'])
    def key(row):
        return row['recording'], Path(row['source']['shard']).name, row['tile']['start_timebin']
    original = {key(r):r for r in read_rows(parent['datasets']['original10k']['path'])}
    for row in rows:
        old = original[key(row)]
        if (row['events'] != old['events'] or row['source'] != old['source']
                or row['tile']['end_timebin'] > old['tile']['end_timebin']):
            raise ValueError('1k labels are not nested unchanged within original10k')
    config = AutoConfig.from_pretrained(parent['backbone'], revision=parent['revision'],
        trust_remote_code=True, local_files_only=True)
    data = PixelWindows(rows, 'data/xcl/shards', config)
    ids = sorted({r['recording'] for r in rows})
    if (sum(w[2] for w in data.windows) != 200000 or len(data) != budget['windows']
            or ids != budget['recording_ids'] or set(ids) & set(validation['recording_ids'])):
        raise ValueError('incorrect budget or recording-disjoint split')
    # Read every selected real input now, detecting missing/corrupt shards before GPU work.
    inputs, masks = hashlib.sha256(), hashlib.sha256()
    for index in range(len(data)):
        audio, target, valid = data[index]
        if not torch.isfinite(audio).all() or not torch.all((target == 0) | (target == 1)):
            raise ValueError('invalid spectrogram or nonbinary training target')
        inputs.update(audio.numpy().tobytes())
        masks.update(target[:, :valid].numpy().tobytes())
    protected = {**parent['protected_files'], str(PARENT / 'manifest.json'):digest(PARENT / 'manifest.json'),
        str(scaling_path):digest(scaling_path), source['path']:source['sha256']}
    for path in PARENT.rglob('*'):
        if path.is_file() and path.suffix in {'.json','.png','.pdf','.svg','.csv'}:
            protected[str(path)] = digest(path)
    train = dict(**source, seconds=1000, windows=len(data), recording_ids=ids,
        supervision=data.supervision_counts(), preflight_input_sha256=inputs.hexdigest(),
        preflight_mask_sha256=masks.hexdigest())
    inherited = ['backbone','revision','seeds','epochs','learning_rate','training_config','anchor',
        'checkpoint_selection','evaluation','inference','caveat','reuse']
    plan = {k:parent[k] for k in inherited}
    plan.update(run=RUN, widths=[128], layers=[1,0], stage_order=['pointwise'],
        datasets={'original1k':train, 'validation':validation}, parent_manifest=str(PARENT / 'manifest.json'),
        selection=scaling['selection'], comparison='six fresh runs: same 1k data and five-epoch budget for both heads',
        pairing='same seed, shared-weight initialization, sample order and CPU RNG; vary transformer block only',
        old_1k_results='three-epoch models are historical context, not controls for this five-epoch comparison',
        protected_files=protected, code_sha256={**parent['code_sha256'], SCRIPT:digest(SCRIPT)})
    verify(plan)
    write_json(manifest, plan)
    return plan


def execute(spec, plan, gpu):
    path = OUT / 'conditions' / f'{spec["name"]}.json'
    study.frozen_json(path, spec)
    if shutil.disk_usage('.').free < 1536 * 1024**2:
        raise ValueError('less than 1.5 GiB free; preserve data and stop')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':gpu, 'HF_HUB_OFFLINE':'1', 'OMP_NUM_THREADS':'4',
        'MKL_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'PYTHONUNBUFFERED':'1'}
    commands = []
    if not Path(spec['checkpoint']).exists():
        commands.append(('train', ['scripts/train.py', '--annotations',plan['datasets']['original1k']['path'],
            '--validation-annotations',plan['datasets']['validation']['path'], '--backbone',plan['backbone'],
            '--backbone-revision',plan['revision'], '--hidden','128', '--head-layers',str(spec['layers']),
            '--seed',str(spec['seed']), '--epochs','5', '--learning-rate','0.001', '--batch-size','16',
            '--eval-batch-size','4', '--accumulation','1', '--dropout','0.1', '--weight-decay','0.0001',
            '--tv-weight','0', '--no-target-smoothing', '--log-every','5', '--out',spec['checkpoint']]))
    commands.append(('evaluate', [SCRIPT, '--evaluate',str(path)]))
    for action, args in commands:
        write_json(OUT / f'gpu{study.GPUS.index(gpu)}.json', dict(state=action, condition=spec['name']))
        print(f'{action}: {spec["name"]} on {gpu}', flush=True)
        with (OUT / 'logs' / f'{action}_{spec["name"]}.log').open('a') as log:
            subprocess.run([sys.executable, '-u', *args], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    study.measure(spec, plan)


def run(plan):
    if not PAUSE.exists() or subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise ValueError('Qwen must remain paused and both GPUs must be free')
    (OUT / 'logs').mkdir(exist_ok=True)
    # Bind output locations only, within this process; reuse the frozen evaluator/summary unchanged.
    study.OUT, study.ARTIFACTS = OUT, ARTIFACTS
    specs = [study.condition(f'layers{layers}',128,seed,dataset='original1k',layers=layers)
        for seed in plan['seeds'] for layers in plan['layers']]
    write_json(OUT / 'status.json',dict(state='running',stage='pointwise',conditions=[s['name'] for s in specs]))
    def lane(index):
        for spec in specs[index::2]:
            execute(spec, plan, study.GPUS[index])
        write_json(OUT / f'gpu{index}.json',dict(state='complete'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane,index) for index in range(2)]
        for future in futures:
            future.result()
    for seed in plan['seeds']:
        pair = [unpack(Path(s['report']))[0] for s in specs if s['seed']==seed]
        for a,b in zip(pair[0]['training_history'],pair[1]['training_history']):
            if any(a[k]!=b[k] for k in ['epoch','sample_order_sha256','cpu_rng_sha256']):
                raise ValueError('paired training order or CPU randomness differs')
    ordered = sorted(specs,key=lambda s:(-s['layers'],s['seed']))
    study.summarize_stage('pointwise',ordered,plan,'layers')
    verify(plan)
    write_json(OUT / 'status.json',dict(state='complete',runs=len(specs),training_seconds=1000))
    print(f'All six runs complete: {OUT}. Qwen remains paused.',flush=True)


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',type=Path)
    args = parser.parse_args()
    plan = prepare()
    if args.prepare:
        print(json.dumps({k:plan['datasets'][k] for k in ['original1k','validation']},indent=2))
    elif args.evaluate:
        evaluator.OUT = OUT
        evaluator.evaluate(args.evaluate)
    else:
        try:
            run(plan)
        except Exception as error:
            write_json(OUT / 'status.json',dict(state='failed',error=str(error)))
            raise


if __name__ == '__main__':
    main()
