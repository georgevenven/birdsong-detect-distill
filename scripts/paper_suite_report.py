#!/usr/bin/env python3
"""Paper-ready, source-linked tables and a square two-panel scaling/backbone figure."""
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import paper_suite as suite
from birdsong_detect_distill.benchmark_data import digest
from prepare_detector_study import verify

LABELS = dict(songmae='SongMAE-Large (ours)',released_n='YOLO11n (released)',
    released_l='YOLO11l (released)',teacher_n='YOLO11n (our teacher labels)',
    teacher_l='YOLO11l (our teacher labels)',birdcode='BirdCODE')


def development(plan):
    folder = suite.OUT/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    jobs = suite.SONGMAE_JOBS
    reports = {job:json.loads((suite.OUT/'runs'/job/'loro.json').read_text()) for job in jobs}
    teacher_path = suite.ROOT/'results/qwen_teacher_powdermill/leave_one_recording_out_2026-09-14/self_review_1.json'
    teacher = json.loads(teacher_path.read_text())
    if any(r['summary']['seconds']!=23100 or r['summary']['segments']!=77 for r in [teacher,*reports.values()]):
        raise ValueError('teacher/student figure coverage differs')
    plt.style.use('default')
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':10,
        'xtick.labelsize':9,'ytick.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,(left,right) = plt.subplots(1,2,sharey=True,figsize=(3.5,3.5))
    fig.subplots_adjust(left=.17,right=.98,bottom=.23,top=.89,wspace=.16)
    budgets = suite.BUDGETS
    ap = [reports[f'large_{n}']['summary']['ap'] for n in budgets]
    left.plot(budgets,ap,color='.55',marker='o',linewidth=1.6,markersize=4)
    left.plot(budgets[-1],ap[-1],color='C0',marker='o',markersize=5)
    left.set_xscale('log')
    left.set_xticks(budgets,labels=['100','1k','10k','15k'],rotation=45,ha='right')
    left.xaxis.set_minor_locator(NullLocator())
    left.set(xlim=(65,22000),ylim=(0,1),ylabel='Pixel AP',xlabel='Training audio (s)',title='(a) Label budget')
    left.set_yticks([0,.2,.4,.6,.8,1])
    sizes = ['micro','base','large']
    values = [reports[f'{s}_15000']['summary']['ap'] for s in sizes]
    bars = right.bar(range(3),values,width=.55,color=['.65','.65','C0'])
    right.bar_label(bars,fmt='%.3f',padding=4,fontsize=8.5)
    right.set_xticks(range(3),labels=[s.title() for s in sizes])
    right.set(xlim=(-.6,2.6),xlabel='Backbone',title='(b) Model size')
    for axis in [left,right]:
        line = axis.axhline(teacher['summary']['ap'],color='.25',linestyle='--',linewidth=1.2)
    left.legend([line],['Qwen_teacher\nlabels'],loc='lower right',frameon=False,fontsize=8.5,handlelength=1.2)
    for suffix in ['png','pdf','svg']:
        fig.savefig(folder/f'scaling_backbones.{suffix}',dpi=600,facecolor='white')
    plt.close(fig)
    sources = {str(suite.OUT/'runs'/job/'loro.json'):digest(suite.OUT/'runs'/job/'loro.json') for job in jobs}
    sources[str(teacher_path)] = digest(teacher_path)
    summary = dict(training_budgets_seconds=budgets,validation_seconds=plan['datasets']['validation']['seconds'],
        seed=0,teacher=teacher['summary'],models={job:r['summary'] for job,r in reports.items()},source_sha256=sources)
    suite.atomic(suite.OUT/'development.json',summary)
    lines = ['# Powdermill development results','',
        'All 77 segments, four original recordings, 23,100 seconds. Leave-one-original-recording-out threshold calibration.',
        f"Single seed (0); five epochs maximum, best XC validation BCE; shared {plan['datasets']['validation']['seconds']:,} s XC validation.",'',
        '| Model | Training seconds | Pixel AP | 2D IoU |','|---|---:|---:|---:|']
    for job in jobs:
        size,n = job.split('_');s = reports[job]['summary']
        lines.append(f'| SongMAE-{size.title()} | {n} | {s["ap"]:.3f} | {s["iou"]:.3f} |')
    lines += ['', 'The Large/15k checkpoint is shared by both figure panels; it is not trained twice.',
        'Figure: `figures/scaling_backbones.png` (also PDF/SVG). No uncertainty bars from a single seed.']
    (suite.OUT/'development.md').write_text('\n'.join(lines)+'\n')


def table(plan, datasets, models, metric, path):
    keys = ['ap','iou','pooled_precision','pooled_recall']
    rows, sources = [], {}
    for model in models:
        row = {'Model':LABELS[model]}
        for dataset in datasets:
            p = suite.OUT/'external'/model/f'{dataset}.json'
            report = json.loads(p.read_text())
            if report['manifest_sha256']!=digest(suite.OUT/'manifest.json') or report['diagnostic_only']:
                raise ValueError('incomplete or mismatched external result')
            if {r['name'] for r in report['per_recording']['evaluation']}!=set(plan['external'][dataset]['recordings']):
                raise ValueError('external table coverage differs')
            cal = suite.OUT/'external'/model/'powdermill.json'
            if report['calibration_sha256']!=digest(cal):
                raise ValueError('external threshold provenance differs')
            value = (report['site_macro'] if dataset=='wabad' else report['summary'])[metric]
            row.update({f'{dataset}_{key}':value[key] for key in keys})
            sources[str(p)] = digest(p)
            sources[str(cal)] = digest(cal)
        rows.append(row)
    for suffix,delimiter in [('csv',','),('tsv','\t')]:
        with path.with_suffix('.'+suffix).open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=delimiter)
            writer.writeheader();writer.writerows(rows)
    columns = ['Model',*[f'{d} {k}' for d in datasets for k in ['AP','IoU']]]
    text = ['| '+' | '.join(columns)+' |','|'+'|'.join(['---']+['---:']*(len(columns)-1))+'|']
    for row in rows:
        text.append('| '+' | '.join([row['Model'],*[f'{row[f"{d}_{k}"]:.3f}' for d in datasets for k in ['ap','iou']]])+' |')
    return text,sources


def main(development_only):
    plan = suite.prepare()
    development(plan)
    if development_only:
        return
    two_d = ['released_n','released_l','teacher_n','teacher_l','songmae']
    temporal = ['birdcode',*two_d]
    table2,sources2 = table(plan,['wabad','hawaii'],two_d,'full',suite.OUT/'table2')
    table3,sources3 = table(plan,['xcsl','nips4bplus'],temporal,'temporal',suite.OUT/'table3')
    lines = ['# Updated external results','','## Table 2 — time–frequency localization','',*table2,
        '','## Table 3 — temporal localization','',*table3,'',
        'XC-AJ is named `xcsl` internally. AP uses continuous scores. IoU thresholds are separately selected per model and task on Powdermill Recordings 2–4, never on external datasets.',
        'WABAD: equal-site macro over the same 68 sites / 4,264 retained clips. Hawaii: recording macro over 635 clips. XC-AJ: the same 288 known-index-disjoint clips. NIPS4Bplus: 674 annotated clips with the same ignored intervals.',
        'CSV/TSV files additionally retain precision and recall. Models keep their native inputs; common 5-ms scoring grid. Full-band area is mel-weighted, based on human box unions, not manually segmented pixels.',
        '','## Draft corrections','',
        f"- This run uses 15,000 s training ({15000/3600:.3f} h) and {plan['datasets']['validation']['seconds']:,} s validation, not 25,000 s training.",
        '- The full SongMAE detection encoder is fine-tuned, including CNN and positional embeddings; those are not frozen.',
        '- Table 2 uses WABAD and Hawaii. XC-AJ is in temporal Table 3; the dataset paragraph currently swaps these.',
        '- Thresholds are model- and task-specific, then fixed for external evaluation—not one threshold shared across every experiment.',
        '- YOLO comparisons include both nano and large, released and teacher-fine-tuned. Released human-trained checkpoints are not retrained on their original human dataset.',
        '- The teacher-trained YOLO and SongMAE share labels but differ in pretraining, image representation and optimization, so differences cannot be attributed only to architecture.',
        '- Figure 3 now reports full-Powdermill cross-calibrated results. The old Table 1 teacher values were Recording_1-only; its scope must be distinguished.',
        '- Rewrite performance claims from these measured results; old claims are not assumed true.',
        '',plan['caveat']]
    (suite.OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    suite.atomic(suite.OUT/'table_sources.json',dict(manifest_sha256=digest(suite.OUT/'manifest.json'),reports_sha256={**sources2,**sources3}))
    verify(plan)
    print(suite.OUT/'tables.md',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-only',action='store_true')
    main(parser.parse_args().development_only)
