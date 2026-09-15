#!/usr/bin/env python3
"""Cross-calibrate existing full-Powdermill scores; no training or inference."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import THRESHOLDS, summarize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/qwen_teacher_powdermill'
OUT = RESULTS / 'leave_one_recording_out_2026-09-14'
GROUPS = {'Recording_1': 36, 'Recording_2': 14, 'Recording_3': 1, 'Recording_4': 26}
EXPECTED = {f'{g}_Segment_{i:02d}' for g, n in GROUPS.items() for i in range(1, n + 1)}
CONDITIONS = [('direct_plain', 'Qwen: no reasoning, no axes'),
    ('direct_axes', 'Qwen: no reasoning, axes'), ('reasoning', 'Qwen: reasoning, axes'),
    ('self_review_1', 'Qwen: reasoning + one self-review'),
    ('self_review_2', 'Qwen: reasoning + two self-reviews')]


def sources():
    for condition, label in CONDITIONS:
        yield dict(id=condition, label=label, family=condition, seed=None, threshold_floor_index=1,
            path=RESULTS / f'prompt_full_2026-09-11/scores/{condition}.json')
    for seed in range(3):
        for family, folder, name, label in [
            ('linear', 'pointwise_bare_linear_2000train_800val', 'bare_linear_d0', 'SongMAE-Large + bare linear'),
            ('transformer', 'new_teacher_2000train_800val', 'layers1_d128', 'SongMAE-Large + transformer')]:
            yield dict(id=f'{family}_s{seed}', label=label, family=family, seed=seed, threshold_floor_index=0,
                path=RESULTS / f'{folder}_2026-09-14/runs/{name}_new2k_s{seed}/comparison.json')


def group(row):
    return row['name'].split('_Segment_')[0]


def checked_rows(report):
    rows = [r for part in report['per_recording'].values() for r in part]
    if len(rows) != 77 or {r['name'] for r in rows} != EXPECTED:
        raise ValueError('duplicate or missing full-Powdermill segments')
    if Counter(group(r) for r in rows) != GROUPS or any(r['seconds'] != 300 for r in rows):
        raise ValueError('inconsistent source grouping or incomplete intervals')
    for row in rows:
        value = row['area']['full']
        counts = np.asarray(value['counts'])
        if counts.shape != (101, 3) or np.any(counts < 0) or np.any(counts != counts.astype(np.int64)):
            raise ValueError('invalid saved confusion counts')
        tp, fp, fn = counts.T
        union = tp + fp + fn
        curve = np.divide(tp, union, out=np.ones(101), where=union != 0)
        if not np.allclose(curve, value['iou_curve'], rtol=0, atol=1e-12):
            raise ValueError('saved IoU disagrees with counts')
        if np.any(tp + fn != tp[0] + fn[0]) or tp[0] + fp[0] != 128 * 60000:
            raise ValueError('reference pixels or evaluated area changed')
        if np.any(np.diff(tp) > 0) or np.any(np.diff(tp + fp) > 0):
            raise ValueError('nonmonotonic threshold counts')
        if (value['ap'] is None) != (tp[0] + fn[0] == 0):
            raise ValueError('inconsistent positive-segment AP policy')
    return sorted(rows, key=lambda r: r['name'])


def evaluate(spec, report, rows):
    folds, scored = [], []
    for held_out in GROUPS:
        calibration = [r for r in rows if group(r) != held_out]
        evaluation = [r for r in rows if group(r) == held_out]
        floor = spec['threshold_floor_index']
        curve = np.mean([r['area']['full']['iou_curve'] for r in calibration], axis=0)
        index = floor + int(np.argmax(curve[floor:]))
        metrics = summarize(evaluation, {'full': index})['full']
        if held_out == 'Recording_1':
            previous = report['evaluation'] if spec['seed'] is None else report['summary']
            if any(not np.isclose(metrics[k], previous[k], rtol=0, atol=1e-12) for k in previous):
                raise ValueError('Recording_1 fold does not reproduce the previous report')
        folds.append(dict(held_out=held_out, calibration_groups=[g for g in GROUPS if g != held_out],
            calibration_segments=[r['name'] for r in calibration], evaluation_segments=[r['name'] for r in evaluation],
            seconds=sum(r['seconds'] for r in evaluation), threshold_index=index,
            calibration_iou=float(curve[index]), **metrics))
        for row in evaluation:
            value = row['area']['full']
            scored.append(dict(name=row['name'], group=held_out, seconds=row['seconds'],
                ap=value['ap'], iou=value['iou_curve'][index], threshold=float(THRESHOLDS[index]),
                counts=value['counts'][index]))
    tp, fp, fn = np.sum([r['counts'] for r in scored], axis=0)
    summary = dict(ap=float(np.mean([r['ap'] for r in scored if r['ap'] is not None])),
        iou=float(np.mean([r['iou'] for r in scored])), positive_segments=sum(r['ap'] is not None for r in scored),
        segments=len(scored), seconds=sum(r['seconds'] for r in scored),
        pooled_iou=float(tp / max(1, tp + fp + fn)), pooled_precision=float(tp / max(1, tp + fp)),
        pooled_recall=float(tp / max(1, tp + fn)), counts=[int(tp), int(fp), int(fn)])
    return dict(**{k: v for k, v in spec.items() if k != 'path'}, source=str(spec['path']),
        previous_recording1=report['evaluation'] if spec['seed'] is None else report['summary'],
        summary=summary, equal_recording_weighted={k:float(np.mean([f[k] for f in folds])) for k in ['ap', 'iou']},
        folds=folds, per_segment=scored)


def main():
    OUT.mkdir(exist_ok=True)
    specs = list(sources())
    protected = {str(s['path']): digest(s['path']) for s in specs}
    protected.update({str(p): digest(p) for p in [Path(__file__),
        ROOT / 'src/birdsong_detect_distill/benchmark_metrics.py',
        RESULTS / 'prompt_full_2026-09-11/manifest.json']})
    protocol = dict(source_sha256=protected, groups=GROUPS, seconds=23100, windows_5s=4620,
        folds='Hold out one entire original recording; calibrate on all five-minute segments from the other three.',
        threshold_selection='Maximum mean segment IoU, lowest threshold breaks exact ties; retain original grids: Qwen 0.01–1.00, students 0.00–1.00, step 0.01.',
        aggregation='Primary: mean segment AP over 76 reference-positive segments; mean IoU over all 77 segments. Secondary: equal weight to four original-recording summaries. Pooled counts also retained.',
        ap='Reuse exact continuous-score pixel AP per segment, unchanged by calibration. No pooled pixel AP is inferred from the 101-point threshold counts.',
        inference='Unchanged cached predictions. Student probability smoothing sigma (2 mel, 3 frames); teacher confidence-rasterized boxes without smoothing.',
        training='Student checkpoints unchanged: 2000 s XC train, 800 s disjoint XC validation, five epochs maximum, best validation BCE; seeds 0,1,2. No retraining or new teacher calls.',
        caveat='Exploratory development-set reanalysis. Architecture, smoothing and Qwen axes/prompts were already selected using Powdermill; this is not nested model-selection CV or an independent generalization test. Recording_3 contains only five minutes.',
        thresholds='Fold-specific thresholds are only for this analysis; existing image and external-evaluation thresholds are not changed.')
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('frozen reanalysis inputs changed')
    write_json(path, protocol)
    runs, references, training = [], None, None
    for spec in specs:
        report = json.loads(spec['path'].read_text())
        rows = checked_rows(report)
        truths = {r['name']:r['area']['full']['counts'][0] for r in rows}
        if references is not None and truths != references:
            raise ValueError('teacher/student reference area or coverage differs')
        references = truths
        if spec['seed'] is None:
            if report['manifest_sha256'] != digest(RESULTS / 'prompt_full_2026-09-11/manifest.json'):
                raise ValueError('teacher coverage manifest changed')
        else:
            saved = report['protocol']['checkpoint']
            keys = ['training_recording_ids', 'validation_recording_ids', 'annotations_sha256',
                    'validation_annotations_sha256', 'backbone_id', 'backbone_revision']
            identity = {k:saved[k] for k in keys}
            if training is not None and training != identity:
                raise ValueError('student training split or backbone differs')
            training = identity
            if saved['seed'] != spec['seed'] or saved['metrics']['train_timebins'] != 400000:
                raise ValueError('unexpected student seed or training budget')
        run = evaluate(spec, report, rows)
        write_json(OUT / f"{spec['id']}.json", run)
        runs.append(run)
    families = []
    for family in dict.fromkeys(r['family'] for r in runs):
        selected = [r for r in runs if r['family'] == family]
        metrics = {k:dict(mean=float(np.mean([r['summary'][k] for r in selected])),
            seed_sd=float(np.std([r['summary'][k] for r in selected], ddof=1)) if len(selected) > 1 else None)
            for k in ['ap', 'iou', 'pooled_precision', 'pooled_recall']}
        families.append(dict(family=family, label=selected[0]['label'], runs=len(selected), **metrics))
    write_json(OUT / 'summary.json', dict(protocol_sha256=digest(path), families=families, runs=runs))
    lines = ['# Full-Powdermill leave-one-recording-out calibration', '',
        'All 4 original recordings, 77 five-minute segments, 4,620 five-second windows, 23,100 seconds.',
        'Each segment is scored once, at the threshold selected exclusively from the other original recordings.',
        'Primary aggregation preserves the previous segment-macro definition. Student ± values are sample SD across three training seeds, not confidence intervals across recordings.', '',
        '| Model / teacher condition | Pixel AP | 2D IoU |', '|---|---:|---:|']
    def formatted(metric):
        return f"{metric['mean']:.3f}" + (f" ± {metric['seed_sd']:.3f}" if metric['seed_sd'] is not None else '')
    for family in families:
        lines.append(f"| {family['label']} | {formatted(family['ap'])} | {formatted(family['iou'])} |")
    lines += ['', '## Per-original-recording scores', '',
        'Students below are seed 0; JSON and CSV include all seeds.', '',
        '| Model / condition | Held-out recording | Minutes | Threshold | Pixel AP | 2D IoU |', '|---|---|---:|---:|---:|---:|']
    for run in runs:
        if run['seed'] not in [None, 0]:
            continue
        for fold in run['folds']:
            lines.append(f"| {run['label']} | {fold['held_out']} | {fold['seconds']/60:g} | {fold['threshold']:.2f} | {fold['ap']:.3f} | {fold['iou']:.3f} |")
    lines += ['', '## Interpretation and limitations', '', protocol['aggregation'], protocol['ap'],
        'Recording_1 folds reproduce every metric in the existing single-recording reports to numerical precision. Changes in overall AP arise from including Recordings 2–4, not from threshold calibration.',
        protocol['caveat'], protocol['threshold_selection'], protocol['thresholds']]
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    with (OUT / 'folds.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'seed', 'held_out', 'seconds', 'threshold',
            'ap', 'iou', 'pooled_precision', 'pooled_recall'])
        writer.writeheader()
        for run in runs:
            for fold in run['folds']:
                writer.writerow(dict(model=run['label'], seed=run['seed'],
                    **{k:fold[k] for k in writer.fieldnames[2:]}))
    if any(digest(p) != sha for p, sha in protected.items()):
        raise ValueError('source changed during reanalysis')
    write_json(OUT / 'complete.json', dict(state='complete', reports=len(runs), folds=4 * len(runs),
        source_reports_unchanged=True, files={p.name:digest(p) for p in OUT.iterdir() if p.name != 'complete.json'}))
    print('\n'.join(lines[:15]))
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
