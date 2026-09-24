#!/usr/bin/env python3
"""Three-seed tables and square scaling/backbone figure; never select a best seed."""
import argparse
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

import paper_25k_suite as suite
from birdsong_detect_distill.benchmark_data import digest
from prepare_detector_study import verify

METRICS = ['ap','iou','pooled_precision','pooled_recall']
FAMILIES = {
    'SongMAE-Large': [f'large_25000_s{s}' for s in suite.SEEDS],
    'YOLO11n (teacher labels)': [f'yolo_n_25000_s{s}' for s in suite.SEEDS],
    'YOLO11l (teacher labels)': [f'yolo_l_25000_s{s}' for s in suite.SEEDS],
    'YOLO11n (released)': ['released_n'],
    'YOLO11l (released)': ['released_l'],
    'BirdCODE': ['birdcode'],
}


def stats(values):
    return dict(mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)


def fmt(value):
    return f'{value["mean"]:.3f}' + (f' ± {value["seed_sd"]:.3f}' if value['seed_sd'] is not None else '')


def csv_file(path,rows):
    if rows:
        with path.open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)


def development(plan):
    rows, groups, sources = [],defaultdict(list),{}
    for job in [*suite.SONGMAE_JOBS,*suite.YOLO_JOBS]:
        family,size,budget,seed = suite.job_parts(job)
        path = suite.OUT/('runs' if family=='songmae' else 'external')/job/'loro.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text()); summary = report['summary']
        if report['seed'] != seed or summary['segments'] != 77 or summary['seconds'] != 23100:
            raise ValueError('incomplete or wrong-seed development result')
        checkpoint_path = suite.OUT/'runs'/job/'checkpoint.json'
        checkpoint = json.loads(checkpoint_path.read_text())
        if family == 'songmae':
            manifest_path = suite.OUT/'runs'/job/'manifest.json'
            if checkpoint['manifest_sha256'] != digest(manifest_path):
                raise ValueError('SongMAE run provenance differs')
            epoch = checkpoint['metrics']['selected_epoch']
        else:
            if checkpoint['manifest_sha256'] != digest(suite.OUT/'manifest.json'):
                raise ValueError('YOLO run provenance differs')
            epoch = checkpoint['selected_epoch']
        row = dict(job=job,family=family,size=size,training_seconds=budget,seed=seed,selected_epoch=epoch,
            **{k:summary[k] for k in METRICS},checkpoint_sha256=checkpoint['sha256'])
        rows.append(row); groups[(family,size,budget)].append(row)
        sources[str(path)] = digest(path); sources[str(checkpoint_path)] = digest(checkpoint_path)
    aggregates = []
    for (family,size,budget), runs in groups.items():
        if {r['seed'] for r in runs} != set(suite.SEEDS):
            continue
        aggregates.append(dict(family=family,size=size,training_seconds=budget,runs=3,
            **{k:stats([r[k] for r in runs]) for k in METRICS}))
    csv_file(suite.OUT/'development_per_seed.csv',rows)
    suite.atomic(suite.OUT/'development.json',dict(manifest_sha256=digest(suite.OUT/'manifest.json'),
        validation_seconds=2500,runs=rows,aggregates=aggregates,source_sha256=sources))
    lines = ['# Powdermill development — 25k study','',
        'All 77 segments (23,100 s), leave-one-original-recording-out operating-threshold selection.',
        'Mean ± sample SD across training seeds 0, 1, 2. A row appears only after all three seeds complete.',
        'Ten epochs; best checkpoint by XC validation BCE (SongMAE) or native validation box mAP50-95 (YOLO).',
        'Fixed 2,500 s XC validation; nested training budgets. Powdermill is not training audio.','',
        '| Model | Training s | Pixel AP | 2D IoU |','|---|---:|---:|---:|']
    for a in aggregates:
        label = 'SongMAE-'+a['size'].title() if a['family']=='songmae' else 'YOLO11'+a['size']
        lines.append(f'| {label} | {a["training_seconds"]} | {fmt(a["ap"])} | {fmt(a["iou"])} |')
    (suite.OUT/'development.md').write_text('\n'.join(lines)+'\n')
    songmae = {(a['size'],a['training_seconds']):a for a in aggregates if a['family']=='songmae'}
    if len(songmae) == 7:
        # Same-seed 25k runs must use the same minibatch order across backbone sizes.
        for seed in suite.SEEDS:
            histories = [json.loads((suite.OUT/'runs'/f'{size}_25000_s{seed}'/'history.json').read_text()) for size in ['micro','base','large']]
            orders = [[r['sample_order_sha256'] for r in history] for history in histories]
            if len(orders[0]) != 10 or any(order != orders[0] for order in orders[1:]):
                raise ValueError('same-seed backbone minibatch orders differ')
        figure(songmae)
    return rows


def figure(values):
    teacher = json.loads((suite.ROOT/'results/qwen_teacher_powdermill/leave_one_recording_out_2026-09-14/self_review_1.json').read_text())['summary']['ap']
    plt.style.use('default')
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9,
        'ytick.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,(left,right) = plt.subplots(1,2,sharey=True,figsize=(3.5,3.5))
    fig.subplots_adjust(left=.17,right=.98,bottom=.23,top=.89,wspace=.16)
    ap = [values[('large',n)]['ap'] for n in suite.BUDGETS]
    left.errorbar(suite.BUDGETS,[a['mean'] for a in ap],yerr=[a['seed_sd'] for a in ap],
        color='.55',marker='o',markersize=4,linewidth=1.5,capsize=2)
    left.plot(25000,ap[-1]['mean'],'o',color='C0',markersize=5)
    left.set_xscale('log'); left.set_xticks(suite.BUDGETS,labels=['100','1k','5k','10k','25k'],rotation=55,ha='right')
    left.xaxis.set_minor_locator(NullLocator())
    left.set(xlim=(65,40000),ylim=(0,1),ylabel='Pixel AP',xlabel='Training audio (s)',title='(a) Label budget')
    left.set_yticks([0,.2,.4,.6,.8,1])
    sizes = ['micro','base','large']; ap = [values[(s,25000)]['ap'] for s in sizes]
    bars = right.bar(range(3),[a['mean'] for a in ap],yerr=[a['seed_sd'] for a in ap],capsize=2,
        width=.55,color=['.65','.65','C0'])
    right.bar_label(bars,fmt='%.3f',padding=5,fontsize=8)
    right.set_xticks(range(3),labels=[s.title() for s in sizes])
    right.set(xlim=(-.6,2.6),xlabel='Backbone',title='(b) Model size')
    for axis in [left,right]:
        line = axis.axhline(teacher,color='.25',linestyle='--',linewidth=1.2)
    left.legend([line],['Qwen_teacher\nlabels'],loc='lower right',frameon=False,fontsize=8,handlelength=1.2)
    folder = suite.OUT/'figures'; folder.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(folder/f'scaling_backbones.{ext}',dpi=600,facecolor='white')
    plt.close(fig)


def external(plan,final):
    reports,rows,sources = {},[],{}
    for jobs in FAMILIES.values():
        for job in jobs:
            for dataset in ['wabad','hawaii','xcsl','nips4bplus']:
                if job == 'birdcode' and dataset in ['wabad','hawaii']:
                    continue
                path = suite.OUT/'external'/job/f'{dataset}.json'
                if not path.exists():
                    if final:
                        raise ValueError('missing final report: '+str(path))
                    continue
                r = json.loads(path.read_text()); cal = path.with_name('powdermill.json')
                if (r['manifest_sha256'] != digest(suite.OUT/'manifest.json') or r['diagnostic_only']
                    or r['calibration_sha256'] != digest(cal)
                    or {x['name'] for x in r['per_recording']['evaluation']} != set(plan['external'][dataset]['recordings'])):
                    raise ValueError('external provenance or coverage differs')
                metric = 'full' if dataset in ['wabad','hawaii'] else 'temporal'
                value = (r['site_macro'] if dataset=='wabad' else r['summary'])[metric]
                reports[(job,dataset)] = value; sources[str(path)] = digest(path)
                sources[str(cal)] = digest(cal)
                rows.append(dict(job=job,dataset=dataset,metric=metric,
                    seed=suite.job_parts(job)[3] if job not in suite.RELEASED else '',
                    **{k:value[k] for k in METRICS}))
    csv_file(suite.OUT/'external_per_seed.csv',rows)
    lines = ['# External comparison — 25k, three seeds','',
        'Mean ± sample SD across all three training seeds, each calibrated separately on Powdermill Recordings 2–4.',
        'Released checkpoints run once; no artificial seed variation. WABAD is site-macro; other datasets are recording-macro.',
        'Missing/incomplete three-seed groups remain pending. No best-seed selection.','']
    aggregates = []
    for title,datasets in [('Table 2 — Pixel AP / 2D IoU',['wabad','hawaii']),('Table 3 — Frame AP / Temporal IoU',['xcsl','nips4bplus'])]:
        lines += ['## '+title,'','| Model | '+' | '.join(f'{d} {m}' for d in datasets for m in ['AP','IoU'])+' |',
                  '|---|'+'---:|'*4]
        for label,jobs in FAMILIES.items():
            if label=='BirdCODE' and datasets[0]=='wabad':
                continue
            cells=[]
            for dataset in datasets:
                if all((job,dataset) in reports for job in jobs):
                    a = dict(model=label,dataset=dataset,runs=len(jobs),
                        **{k:stats([reports[(job,dataset)][k] for job in jobs]) for k in METRICS})
                    aggregates.append(a); cells += [fmt(a[k]) for k in ['ap','iou']]
                else:
                    cells += ['Pending','Pending']
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines.append('')
    (suite.OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    suite.atomic(suite.OUT/'external_summary.json',dict(aggregates=aggregates,source_sha256=sources,
        manifest_sha256=digest(suite.OUT/'manifest.json'),complete=final))


def main(phase):
    plan = suite.prepare(); rows = development(plan)
    if len([r for r in rows if r['family']=='songmae']) != 21:
        raise ValueError('SongMAE phase incomplete')
    if phase == 'final' and len(rows) != 51:
        raise ValueError('training/development runs incomplete')
    if phase != 'songmae':
        external(plan,phase=='final')
    verify(plan)
    print('Reports written:',suite.OUT,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['songmae','partial','final'],required=True)
    main(parser.parse_args().phase)
