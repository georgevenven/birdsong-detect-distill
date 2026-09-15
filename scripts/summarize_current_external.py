#!/usr/bin/env python3
"""Publish matched tables using current students and immutable released-model results."""
import argparse
import csv
import json
from pathlib import Path

from birdsong_detect_distill.benchmark_data import digest
from evaluate_current_models import atomic_json


LABELS = {'birdbox':'YOLO11n (released)', 'qwen_yolo':'YOLO11n (our teacher labels)',
    'birdcode':'BirdCODE', 'songmae':'SongMAE-Large (ours)'}
METRICS = ('ap','iou','pooled_precision','pooled_recall')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--released-large', type=Path, help='Additional completed released YOLO11l run.')
    parser.add_argument('--out', type=Path, help='New table directory; defaults to --results.')
    args = parser.parse_args()
    out = args.out or args.results
    if any((out / name).exists() for name in ['tables.md', 'summary.json', 'table_sources.json',
            'two_dimensional.csv', 'two_dimensional.tsv', 'temporal.csv', 'temporal.tsv']):
        raise ValueError('published tables already exist')
    manifest_path = args.results / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    old = Path('results/competitors_external_2026-09-08')
    old_sources = json.loads((old / 'table_sources.json').read_text())
    reports, provenance = {}, {}
    for model in LABELS:
        current = model in ['qwen_yolo','songmae']
        cal_path = (args.results / 'calibration' if current else Path('results/baselines_matched_2026-09-07/powdermill_final')) / f'{model}.json'
        cal = json.loads(cal_path.read_text())
        if set(r['name'] for r in cal['per_recording']['calibration']) != set(manifest['powdermill']['calibration']):
            raise ValueError('models use different calibration recordings')
        for dataset in (['xcsl','nips4bplus'] if model == 'birdcode' else ['wabad','hawaii','xcsl','nips4bplus']):
            path = (args.results if current else old) / f'{model}_{dataset}.json'
            r = json.loads(path.read_text())
            if r['diagnostic_only'] or r['threshold_indices'] != cal['threshold_indices'] or r['calibration_sha256'] != digest(cal_path):
                raise ValueError(f'non-final or uncalibrated result: {path}')
            if current:
                if r['protocol'] != cal['protocol'] or r['protocol']['manifest_sha256'] != digest(manifest_path):
                    raise ValueError('current inference or split mismatch')
            elif digest(path) != old_sources['reports_sha256'][path.name] or r['inference'] != cal['inference']:
                raise ValueError('released-model result changed')
            expected = manifest['external'][dataset]['recordings']
            if sorted(x['name'] for x in r['per_recording']['evaluation']) != sorted(expected):
                raise ValueError('different external coverage')
            reports[model,dataset] = r
            provenance[str(path)] = digest(path)
    if args.released_large:
        from birdsong_detect_distill.birdbox import MODELS
        if digest(args.released_large / 'manifest.json') != digest(manifest_path):
            raise ValueError('large YOLO uses a different evaluation manifest')
        cal_path = args.released_large / 'calibration/birdbox.json'
        cal = json.loads(cal_path.read_text())
        if (cal['diagnostic_only'] or cal['dataset'] != 'powdermill'
                or cal['protocol']['inference']['checkpoint_sha256'] != MODELS['yolo11l'][0]
                or cal['protocol']['manifest_sha256'] != digest(manifest_path)
                or set(r['name'] for r in cal['per_recording']['calibration']) != set(manifest['powdermill']['calibration'])):
            raise ValueError('large YOLO calibration or checkpoint mismatch')
        provenance[str(cal_path)] = digest(cal_path)
        for dataset, spec in manifest['external'].items():
            path = args.released_large / f'birdbox_{dataset}.json'
            r = json.loads(path.read_text())
            if (r['diagnostic_only'] or r['dataset'] != dataset or r['model'] != 'birdbox'
                    or r['protocol'] != cal['protocol'] or r['threshold_indices'] != cal['threshold_indices']
                    or r['calibration_sha256'] != digest(cal_path)
                    or sorted(x['name'] for x in r['per_recording']['evaluation']) != sorted(spec['recordings'])):
                raise ValueError('large YOLO result is incomplete or differs from calibration')
            reports['birdbox_large',dataset] = r
            provenance[str(path)] = digest(path)
    for dataset in manifest['external']:
        reference = reports['birdbox',dataset]
        expected = {r['name']:r for r in reference['per_recording']['evaluation']}
        for (model,name),r in reports.items():
            if name != dataset:
                continue
            if r['source_audio_sha256'] != reference['source_audio_sha256'] or r['evaluated_seconds'] != reference['evaluated_seconds']:
                raise ValueError('different external audio or scoring duration')
            for row in r['per_recording']['evaluation']:
                other = expected[row['name']]
                if row['seconds'] != other['seconds'] or row['group'] != other['group']:
                    raise ValueError('different recording duration or site')
                for metric,value in row['area'].items():
                    if [tp+fn for tp,fp,fn in value['counts']] != [tp+fn for tp,fp,fn in other['area'][metric]['counts']]:
                        raise ValueError('different reference masks')
    specs = [('two_dimensional',2,('wabad','hawaii'),('WABAD','Hawaii'),'full',
        ('Pixel AP','2D IoU','Pixel precision','Pixel recall'),('birdbox','qwen_yolo','songmae')),
        ('temporal',3,('xcsl','nips4bplus'),('XC-AJ','NIPS4Bplus'),'temporal',
        ('Frame AP','Temporal IoU','Frame precision','Frame recall'),('birdcode','birdbox','qwen_yolo','songmae'))]
    labels_by_model = {**LABELS, 'birdbox_large':'YOLO11l (released)'}
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# Matched external evaluation: current students and released baselines','']
    for name, number, datasets, display, metric, labels, models in specs:
        if args.released_large:
            models = list(models)
            models.insert(models.index('birdbox') + 1, 'birdbox_large')
        rows = []
        for model in models:
            values = []
            for dataset in datasets:
                report = reports[model,dataset]
                summary = report['site_macro' if dataset=='wabad' else 'summary'][metric]
                values.extend(summary[k] for k in METRICS)
            rows.append([labels_by_model[model],*values])
        header = ['Model',*[f'{d} {m}' for d in display for m in labels]]
        for suffix, delimiter in [('.csv',','),('.tsv','\t')]:
            with (out / (name+suffix)).open('w') as stream:
                writer = csv.writer(stream,delimiter=delimiter)
                writer.writerow(header)
                writer.writerows(rows)
        keep = [0,1,2,5,6]
        lines.extend([f'## Table {number}','',
            '| '+' | '.join(header[i] for i in keep)+' |','|---|---:|---:|---:|---:|'])
        lines.extend('| '+' | '.join(f'{row[i]:.3f}' if isinstance(row[i],float) else row[i] for i in keep)+' |' for row in rows)
        lines.append('')
    lines.extend(['Both current students use identical self-reviewed Qwen labels: 10,000 s of XC training audio and 800 s of recording-disjoint XC validation audio.',
        'SongMAE uses hard-mask BCE, Gaussian probability smoothing and a single threshold. YOLO retains its native box objective, spectrogram input, initialization and validation mAP checkpoint selection.',
        'Each model’s pixel and frame thresholds are independently calibrated on the same 41 Powdermill segments from Recordings 2–4, then frozen before external evaluation.',
        'WABAD: 4,264 recordings across 68 sites, site-macro scores. Hawaii: 635 recordings, recording-macro scores. Pixel masks use 128 mel bins over 20–16,000 Hz.',
        'XC-AJ: the frozen 288-recording known-index-disjoint subset. NIPS4Bplus: 674 annotated clips, non-birds negative, Unknown intervals ignored. Temporal scores use the common 5-ms grid.',
        'AP uses continuous scores; IoU uses fixed calibrated thresholds. These are area/occupancy metrics, not event-matching AP. CSV/TSV files include full-precision precision and recall.',
        'Released YOLO and BirdCODE results are reused unchanged after coverage, source-hash, reference-mask and calibration checks. Broader pretraining exposure is not fully certified.',''])
    if args.released_large:
        lines.extend(['YOLO11l is the released BirdBox large checkpoint, without teacher-label retraining. It uses native BirdBox preprocessing, NMS and the same Powdermill calibration recordings; all earlier model results remain unchanged.', ''])
    (out / 'tables.md').write_text('\n'.join(lines))
    keys = ['dataset','model','segments','evaluated_seconds','summary','site_macro','threshold_indices','calibration_sha256']
    atomic_json(out / 'summary.json',{f'{m}_{d}':{k:r[k] for k in keys if k in r} for (m,d),r in reports.items()})
    atomic_json(out / 'table_sources.json',dict(reports_sha256=provenance,manifest_sha256=digest(manifest_path),
        table_generator_sha256=digest(__file__),checks='same recordings, source hashes, intervals, site groups, reference positives and calibration split'))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
