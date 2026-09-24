#!/usr/bin/env python3
"""Freeze the ordered width, expanded-label and pointwise-head study."""
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoConfig

from audit_overlap import zip_ids
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.data import PixelWindows, read_rows
from birdsong_detect_distill.qwen import map_final, read_done
from evaluate_current_backbones import ANCHOR, REVISIONS, DURATION_RESULTS, LARGE_WIDTH_RESULTS, WIDTH_RESULTS
from summarize_three_seed_figures import unpack

RUN = 'detector_study_2026-09-11'
OUT = Path('results/qwen_teacher_powdermill') / RUN
ARTIFACTS = Path('artifacts') / RUN
LABELS = Path('data/annotations/xcl') / RUN
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
CHANGED = ['scripts/train.py', 'scripts/evaluate_current_backbones.py', 'src/birdsong_detect_distill/model.py']
CODE = [*CHANGED, 'scripts/prepare_detector_study.py', 'scripts/run_detector_study.py',
    'scripts/evaluate_detector_study.py', 'scripts/evaluate_2d.py', 'scripts/evaluate_songmae_smoothing.py',
    'scripts/summarize_three_seed_figures.py', 'src/birdsong_detect_distill/data.py',
    'src/birdsong_detect_distill/benchmark_data.py', 'src/birdsong_detect_distill/benchmark_metrics.py',
    'src/birdsong_detect_distill/qwen.py']


def verify(plan):
    for path, sha in {**plan['protected_files'], **plan['code_sha256']}.items():
        if digest(path) != sha:
            raise ValueError(f'frozen study dependency changed: {path}')


def prefix(rows, bins):
    selected = []
    for original in rows:
        if not bins:
            break
        row = copy.deepcopy(original)
        tile = row['tile']
        start = tile['start_timebin']
        size = min(bins, tile['end_timebin'] - start)
        tile.update(end_timebin=start + size, ownership_end_timebin=start + size,
            offset_ms=(start + size) * 5, ownership_offset_ms=(start + size) * 5)
        selected.append(row)
        bins -= size
    if bins:
        raise ValueError('insufficient label duration')
    return selected


def expanded_labels(split):
    master = Path('data/annotations/xcl/qwen38_adaptive_review_5s_annotations.jsonl')
    accepted, latest = read_done(master), {}
    for row in map(json.loads, master.open()):
        if row.get('status') == 'ok':
            tile = row['tile']
            key = row['recording'], tile['ownership_start_timebin'], tile['ownership_end_timebin']
            if key in accepted:
                latest[key] = row
    validation = set(split['partitions']['validation']['recording_ids'])
    excluded = zip_ids(RAW / 'xcaj/Audio.zip') | validation
    groups, intervals, sources = defaultdict(list), defaultdict(list), {}
    for (name, start, end), row in sorted(latest.items()):
        if name.upper() in excluded:
            continue
        stage = [p for p in row['passes'] if p['stage'] == 'Primary self-review']
        if len(stage) != 1 or not 0 <= start < end <= row['source']['end'] - row['source']['start']:
            raise ValueError('invalid self-review stage or source bounds')
        tile = row['tile']
        events = map_final(stage[0]['events'], tile['start_timebin'], tile['end_timebin'], start, end, 128, 5)
        provenance = row.get('config_sha256', 'legacy_1024_reasoning_2048_output')
        item = dict(type='annotation', status='ok', variant='self_review', recording=name,
            source=row['source'], teacher_view=tile, teacher_config=provenance, events=events,
            tile=dict(start_timebin=start, end_timebin=end, ownership_start_timebin=start,
                ownership_end_timebin=end, onset_ms=start * 5, offset_ms=end * 5,
                ownership_onset_ms=start * 5, ownership_offset_ms=end * 5))
        groups[provenance].append(item)
        intervals[name].append((start, end))
        source = (Path(row['source']['shard']).name, row['source']['start'], row['source']['end'])
        if name in sources and sources[name] != source:
            raise ValueError('recording maps to inconsistent source audio')
        sources[name] = source
    for values in intervals.values():
        values.sort()
        if any(end > start for (_, end), (start, _) in zip(values, values[1:])):
            raise ValueError('overlapping ownership windows')
    duration = {key:sum(r['tile']['end_timebin'] - r['tile']['start_timebin'] for r in rows) for key, rows in groups.items()}
    total = sum(duration.values())
    if total < 20000 * 200:
        raise ValueError('fewer than 20,000 eligible self-reviewed seconds')
    # Allocate in two-bin units so the nested 10k condition has identical provenance fractions.
    units = {key:2000000 * value // total for key, value in duration.items()}
    remainder = 2000000 - sum(units.values())
    ranked = sorted(units, key=lambda k: (-(2000000 * duration[k] % total), k))
    for key in ranked[:remainder]:
        units[key] += 1
    outputs = {10000:[], 20000:[]}
    for key, rows in sorted(groups.items()):
        random.Random(17).shuffle(rows)
        for seconds, multiplier in [(10000, 1), (20000, 2)]:
            outputs[seconds].extend(prefix(rows, units[key] * multiplier))
    paths = {}
    for seconds, rows in outputs.items():
        path = LABELS / f'self_review_s{seconds}.jsonl'
        if path.exists():
            raise ValueError('expanded labels already exist without a complete frozen manifest')
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(type='metadata', variant='self_review', training_seconds=seconds,
            source_sha256=digest(master), selection='seed-17 nested windows, matched teacher-provenance proportions')
        path.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in [metadata, *rows]))
        paths[seconds] = path
    small = {(r['recording'], r['tile']['start_timebin']):r for r in outputs[10000]}
    large = {(r['recording'], r['tile']['start_timebin']):r for r in outputs[20000]}
    for key, row in small.items():
        if key not in large or row['events'] != large[key]['events'] or row['tile']['end_timebin'] > large[key]['tile']['end_timebin']:
            raise ValueError('expanded subsets are not nested with identical labels')
    return paths, dict(master=str(master), master_sha256=digest(master), eligible_seconds=total / 200,
        available_seconds_by_provenance={k:v / 200 for k,v in duration.items()},
        selected_seconds_by_provenance={str(s):{k:n * (s // 10000) / 200 for k,n in units.items()} for s in outputs},
        later_stages='Only accepted windows are used; self-review labels extracted before shifted review/adjudication',
        xcaj_exclusion='all XC-AJ recording IDs excluded', validation_recordings=sorted(validation),
        note='Fresh paired 10k/20k training conditions; old 10k training labels are not substituted')


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        verify(plan)
        return plan
    anchor = json.loads(ANCHOR.read_text())['protocol']
    split = json.loads(Path(anchor['split']).read_text())
    protected = {str(ANCHOR):digest(ANCHOR), anchor['split']:anchor['split_sha256']}
    for part in split['partitions'].values():
        source = part['files']['self_review']
        protected[source['path']] = source['sha256']
    old_lr = json.loads(Path('results/qwen_teacher_powdermill/learning_rate_10000train_800val_2026-09-10/manifest.json').read_text())
    for source in CHANGED:
        protected[str(OUT / 'source_before' / source)] = old_lr['code_sha256'][source]
    for source, sha in old_lr['code_sha256'].items():
        if source not in CHANGED:
            protected[source] = sha
    reuse = {}
    for width in (128, 384, 768):
        for seed in (0, 1, 2):
            report = (DURATION_RESULTS / f'seed{seed}/large/comparison.json' if width == 128 else
                LARGE_WIDTH_RESULTS / 'large_d384/comparison.json' if width == 384 else WIDTH_RESULTS / 'large_d768/comparison.json')
            if width != 128 and seed:
                continue
            saved, rows, score, _ = unpack(report)
            history = saved['training_history'][:5]
            if (len(history) != 5 or saved['seed'] != seed or saved['hidden'] != width
                    or saved['metrics']['selected_epoch'] != min(history, key=lambda r:r['validation_loss'])['epoch']):
                raise ValueError('reused checkpoint is not the requested best-of-five condition')
            reuse[f'w{width}_s{seed}'] = dict(report=str(report), checkpoint=saved['checkpoint'],
                report_sha256=digest(report), checkpoint_sha256=saved['checkpoint_sha256'])
            protected[str(report)] = digest(report)
            protected[saved['checkpoint']] = saved['checkpoint_sha256']
    # Protect published figures/tables without copying or rewriting them.
    for folder in [OUT.parent, Path('results/current_external_2026-09-09'), Path('results/yolo11l_external_2026-09-10/tables')]:
        for p in folder.rglob('*'):
            if p.is_file() and OUT not in p.parents and (p.suffix in {'.png','.pdf','.svg','.csv','.tsv'} or p.name == 'tables.md'):
                protected[str(p)] = digest(p)
    for p, sha in protected.items():
        if digest(p) != sha:
            raise ValueError(f'preparation dependency changed: {p}')
    expanded, label_provenance = expanded_labels(split)
    protected[label_provenance['master']] = label_provenance['master_sha256']
    train = Path(split['partitions']['train']['files']['self_review']['path'])
    validation = Path(split['partitions']['validation']['files']['self_review']['path'])
    config = AutoConfig.from_pretrained('georgeven/songmae-large-32x1', revision=REVISIONS['large'],
        trust_remote_code=True, local_files_only=True)
    datasets = {}
    for name, source, seconds in [('original10k',train,10000), ('expanded10k',expanded[10000],10000),
            ('expanded20k',expanded[20000],20000), ('validation',validation,800)]:
        rows = read_rows(source)
        data = PixelWindows(rows, 'data/xcl/shards', config)
        if sum(w[2] for w in data.windows) != seconds * 200:
            raise ValueError('training loader changes the exact audio budget')
        ids = sorted({r['recording'] for r in rows})
        if name != 'validation' and set(ids) & set(split['partitions']['validation']['recording_ids']):
            raise ValueError('training/validation overlap')
        datasets[name] = dict(path=str(source), sha256=digest(source), seconds=seconds,
            windows=len(data), recording_ids=ids, supervision=data.supervision_counts())
        protected[str(source)] = digest(source)
        del data
    plan = dict(run=RUN, backbone='georgeven/songmae-large-32x1', revision=REVISIONS['large'],
        seeds=[0,1,2], widths=[128,384,768], epochs=5, learning_rate=.001,
        training_config=dict(optimizer='AdamW', learning_rate=.001, weight_decay=.0001, batch_size=16,
            accumulation=1, train_all=False, target_smoothing_mel_time=None),
        reuse=reuse, datasets=datasets, label_provenance=label_provenance, anchor=str(ANCHOR),
        stage_order=['width','budget','pointwise'],
        width_selection='highest three-seed mean Powdermill pixel AP; exact ties prefer narrower width',
        budget='selected width; fresh nested 10k/20k, three seeds, five epochs each (20k has more optimizer updates)',
        pointwise='selected width, original10k, three seeds; remove transformer block only; shared weights initialized identically',
        checkpoint_selection='minimum validation BCE over the fixed 800 s XC set within five epochs',
        evaluation='same full Powdermill Recording_1 and calibration Recordings_2–4; no external scoring',
        inference='probability Gaussian sigma=(2 mel, 3 frames); single calibrated threshold; no morphology',
        protected_files=protected, code_sha256={p:digest(p) for p in CODE},
        caveat='exploratory development experiments; seed SD is not a dataset-sampling confidence interval')
    write_json(path, plan)
    return plan


if __name__ == '__main__':
    plan = prepare()
    print(json.dumps({k:v for k,v in plan.items() if k in ['stage_order','reuse','label_provenance','width_selection']}, indent=2))
