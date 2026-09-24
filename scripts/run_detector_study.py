#!/usr/bin/env python3
"""Run three ordered, resumable Powdermill experiments independently of the terminal."""
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import calibrate, summarize
from prepare_detector_study import OUT, ARTIFACTS, prepare, verify
from summarize_three_seed_figures import unpack

GPUS = ['GPU-93189925-4747-5857-e01f-d44d77ad07a7', 'GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556']


def frozen_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'frozen condition changed: {path}')
    if not path.exists():
        write_json(path, value)


def condition(stage, width, seed, dataset='original10k', layers=1, reuse=None):
    name = f'{stage}_d{width}_{dataset}_s{seed}'
    return dict(name=name, stage=stage, width=width, seed=seed, dataset=dataset, layers=layers,
        checkpoint=reuse['checkpoint'] if reuse else str(ARTIFACTS / f'{name}.pt'),
        report=reuse['report'] if reuse else str(OUT / 'runs' / name / 'comparison.json'), reused=bool(reuse))


def execute(spec, plan, gpu):
    spec_path = OUT / 'conditions' / f'{spec["name"]}.json'
    frozen_json(spec_path, spec)
    if spec['reused']:
        return
    if shutil.disk_usage('.').free < 1536 * 1024**2:
        raise ValueError('less than 1.5 GiB free; preserve data and stop')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':gpu, 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
        'OPENBLAS_NUM_THREADS':'4', 'HF_HUB_OFFLINE':'1', 'PYTHONUNBUFFERED':'1'}
    train = plan['datasets'][spec['dataset']]
    commands = []
    if not Path(spec['checkpoint']).exists():
        commands.append(('train', ['scripts/train.py', '--annotations',train['path'], '--validation-annotations',plan['datasets']['validation']['path'],
            '--backbone',plan['backbone'], '--backbone-revision',plan['revision'], '--hidden',str(spec['width']),
            '--head-layers',str(spec['layers']), '--seed',str(spec['seed']), '--epochs','5', '--learning-rate','0.001',
            '--batch-size','16', '--eval-batch-size','4', '--accumulation','1', '--dropout','0.1',
            '--weight-decay','0.0001', '--tv-weight','0', '--no-target-smoothing', '--log-every','25', '--out',spec['checkpoint']]))
    commands.append(('evaluate', ['scripts/evaluate_detector_study.py', str(spec_path)]))
    for action, args in commands:
        print(f'{action}: {spec["name"]} on {gpu}', flush=True)
        with (OUT / 'logs' / f'{action}_{spec["name"]}.log').open('a') as log:
            subprocess.run([sys.executable, '-u', *args], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def measure(spec, plan):
    saved, rows, score, protocol = unpack(Path(spec['report']))
    actual = torch.load(saved['checkpoint'], map_location='cpu', weights_only=True)
    if digest(saved['checkpoint']) != saved['checkpoint_sha256'] or saved['metrics'] != actual['metrics'] or saved['training_history'] != actual['training_history']:
        raise ValueError('checkpoint/report mismatch')
    data, val = plan['datasets'][spec['dataset']], plan['datasets']['validation']
    expected = dict(backbone_id=plan['backbone'], backbone_revision=plan['revision'], hidden=spec['width'],
        seed=spec['seed'], target_smoothing=False, tv_weight=0, dropout=.1, training_config=plan['training_config'],
        annotations_sha256=data['sha256'], validation_annotations_sha256=val['sha256'],
        training_recording_ids=data['recording_ids'], validation_recording_ids=val['recording_ids'])
    for k,v in expected.items():
        if actual[k] != v or saved[k] != v:
            raise ValueError(f'unmatched condition: {spec["name"]}/{k}')
    if actual.get('head_layers',1) != spec['layers'] or actual.get('confidence_targets',False) or actual.get('ignore_uncertain',False):
        raise ValueError('unexpected architecture or target policy')
    history = saved['training_history'][:5]
    if len(history) != 5 or saved['metrics']['selected_epoch'] != min(history,key=lambda r:r['validation_loss'])['epoch']:
        raise ValueError('checkpoint is not best within the five-epoch budget')
    for k,v in dict(train_timebins=data['seconds'] * 200, validation_timebins=160000,
            train_windows=data['windows'], validation_windows=val['windows']).items():
        if saved['metrics'][k] != v:
            raise ValueError('unmatched data budget')
    _, expected_rows, _, reference = unpack(Path(plan['reuse']['w384_s0']['report']))
    for key in ['partitions','student_inference','coordinate_space','mask_rule','threshold_selection','threshold_grid','aggregation','ap_method']:
        if protocol[key] != reference[key]:
            raise ValueError(f'changed evaluation protocol: {key}')
    for part, originals in expected_rows.items():
        if len(rows[part]) != len(originals):
            raise ValueError('incomplete Powdermill coverage')
        for row, old in zip(rows[part], originals):
            if (any(row[k] != old[k] for k in ['name','group','seconds'])
                    or [tp+fn for tp,fp,fn in row['area']['full']['counts']] != [tp+fn for tp,fp,fn in old['area']['full']['counts']]):
                raise ValueError('different reference masks or intervals')
    if summarize(rows['evaluation'],calibrate(rows['calibration']))['full'] != score:
        raise ValueError('metrics or calibration do not reproduce')
    if not spec['reused']:
        if protocol['code_sha256'] != plan['code_sha256'] or protocol['condition'] != spec:
            raise ValueError('new report uses a different implementation or condition')
        cache = Path(spec['checkpoint']).parent / 'predictions' / Path(spec['checkpoint']).stem
        for path in Path(spec['report']).parent.joinpath('segments').glob('*.json'):
            item = json.loads(path.read_text())
            if item['prediction_sha256'] and digest(cache / f'{path.stem}.npz') != item['prediction_sha256']:
                raise ValueError('retained prediction changed')
    positive = [r for r in rows['evaluation'] if r['area']['full']['ap'] is not None]
    return dict(**score, seed=spec['seed'], width=spec['width'], layers=spec['layers'],
        dataset=spec['dataset'], selected_epoch=saved['metrics']['selected_epoch'],
        checkpoint=saved['checkpoint'], checkpoint_sha256=saved['checkpoint_sha256'],
        report=spec['report'], report_sha256=digest(spec['report']),
        parameters=sum(v.numel() for v in actual['head'].values()),
        positive_only_iou_diagnostic=summarize(positive,calibrate(rows['calibration']))['full']['iou'])


def summarize_stage(stage, specs, plan, group_key):
    measured = [measure(spec,plan) for spec in specs]
    groups = {}
    for key in dict.fromkeys(str(row[group_key]) for row in measured):
        values = [r for r in measured if str(r[group_key]) == key]
        if sorted(v['seed'] for v in values) != [0,1,2]:
            raise ValueError('each comparison needs all three seeds')
        groups[key] = dict(per_seed=values, ap_mean=float(np.mean([v['ap'] for v in values])),
            ap_sd=float(np.std([v['ap'] for v in values],ddof=1)), iou_mean=float(np.mean([v['iou'] for v in values])),
            iou_sd=float(np.std([v['iou'] for v in values],ddof=1)))
    report = dict(stage=stage, groups=groups, manifest_sha256=digest(OUT / 'manifest.json'),
        uncertainty='sample SD over fixed-data training seeds, not a dataset confidence interval')
    folder = OUT / stage
    folder.mkdir(exist_ok=True)
    frozen_json(folder / 'summary.json',report)
    table = folder / 'comparison.csv'
    if not table.exists():
        with table.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow([group_key,'Pixel AP mean','Pixel AP SD','2D IoU mean','2D IoU SD'])
            writer.writerows([key,*(v[k] for k in ['ap_mean','ap_sd','iou_mean','iou_sd'])] for key,v in groups.items())
    plot(stage, groups, folder, group_key)
    print(json.dumps({k:{m:v[m] for m in ['ap_mean','ap_sd','iou_mean','iou_sd']} for k,v in groups.items()},indent=2),flush=True)
    return report


def plot(stage, groups, folder, xlabel):
    if (folder / 'pixel_ap.png').exists():
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax = plt.subplots(figsize=(3.5,3.5))
    fig.subplots_adjust(left=.2,right=.97,bottom=.2,top=.9)
    labels = list(groups)
    if stage == 'budget': labels = ['10k s','20k s']
    if stage == 'pointwise': labels = ['1 layer','0 layers']
    ax.bar(labels,[v['ap_mean'] for v in groups.values()],yerr=[v['ap_sd'] for v in groups.values()],capsize=4,color=['.6']*(len(groups)-1)+['C0'])
    ax.set(ylim=(0,1),ylabel='Powdermill pixel AP',xlabel=xlabel,title='Frozen SongMAE-Large')
    fig.text(.58,.035,'3 seeds · best of 5 epochs',ha='center',fontsize=9)
    for suffix in ['png','pdf','svg']: fig.savefig(folder / f'pixel_ap.{suffix}',dpi=600,facecolor='white')
    plt.close(fig)


def run_stage(stage, specs, plan):
    verify(plan)
    print(f'Starting stage: {stage}',flush=True)
    write_json(OUT / 'status.json',dict(stage=stage,state='running',conditions=[s['name'] for s in specs]))
    def lane(index):
        for spec in specs[index::2]: execute(spec,plan,GPUS[index])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,index) for index in range(2)]
        for future in futures: future.result()


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    if not Path('/home/george-vengrovski/.config/birdsong-detect-distill/qwen.paused').exists():
        raise ValueError('Qwen manual pause marker is required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPUs are occupied; do not interfere')
    plan=prepare()
    (OUT / 'logs').mkdir(exist_ok=True)
    width_specs=[condition('width',w,s,reuse=plan['reuse'].get(f'w{w}_s{s}')) for w in plan['widths'] for s in plan['seeds']]
    run_stage('width',width_specs,plan)
    width=summarize_stage('width',width_specs,plan,'width')
    winner=int(min(width['groups'],key=lambda k:(-width['groups'][k]['ap_mean'],int(k))))
    frozen_json(OUT / 'selected_width.json',dict(width=winner,rule=plan['width_selection'],
        summary_sha256=digest(OUT / 'width/summary.json')))
    budget_specs=[condition('budget',winner,s,dataset=d) for d in ['expanded10k','expanded20k'] for s in plan['seeds']]
    run_stage('budget',budget_specs,plan)
    summarize_stage('budget',budget_specs,plan,'dataset')
    controls=[s for s in width_specs if s['width']==winner]
    pointwise=[condition('pointwise',winner,s,layers=0) for s in plan['seeds']]
    run_stage('pointwise',pointwise,plan)
    for new,old in zip(pointwise,controls):
        new_saved=unpack(Path(new['report']))[0]
        old_saved=unpack(Path(old['report']))[0]
        for a,b in zip(new_saved['training_history'],old_saved['training_history'][:5]):
            if any(a[k]!=b[k] for k in ['epoch','sample_order_sha256','cpu_rng_sha256']):
                raise ValueError('pointwise/control training order or CPU randomness differs')
    summarize_stage('pointwise',controls+pointwise,plan,'layers')
    verify(plan)
    write_json(OUT / 'status.json',dict(state='complete',stages=plan['stage_order'],selected_width=winner))
    print(f'All three experiments complete: {OUT}. Qwen remains paused.',flush=True)


if __name__ == '__main__':
    main()
