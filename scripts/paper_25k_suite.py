#!/usr/bin/env python3
"""Detached, three-seed 25k study; reuse the validated full-encoder training path."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoConfig

import paper_suite as older
import run_full_encoder_backbones_snapshot as trainer
from birdsong_detect_distill.benchmark_data import digest
from birdsong_detect_distill.data import PixelWindows, load_spec_slice, read_rows
from prepare_detector_study import verify
from run_qwen_xc_50k import validate
from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate

ROOT, RAW, QUEUE = older.ROOT, older.RAW, older.QUEUE
RUN = 'paper_25000train_3seeds_10epochs_2026-09-15'
ART = older.ART.parent / RUN
OUT = ROOT / 'results' / RUN
LABELS = ROOT / 'data/annotations/xcl' / RUN
COMPLETION = ROOT / 'results/qwen_xc_budget_27500_2026-09-15/completed.json'
BUDGETS, SEEDS = [100, 1000, 5000, 10000, 25000], [0, 1, 2]
SONGMAE_JOBS = [f'large_{n}_s{s}' for n in BUDGETS for s in SEEDS]
SONGMAE_JOBS += [f'{size}_25000_s{s}' for size in ['micro', 'base'] for s in SEEDS]
YOLO_JOBS = [f'yolo_{v}_{n}_s{s}' for n in BUDGETS for v in ['n', 'l'] for s in SEEDS]
EXTERNAL_JOBS = [f'large_25000_s{s}' for s in SEEDS] + [f'yolo_{v}_25000_s{s}' for v in ['n', 'l'] for s in SEEDS]
RELEASED = ['released_n', 'released_l', 'birdcode']
atomic, frozen = older.atomic, older.frozen


def status(state, **values):
    atomic(OUT / 'status.json', dict(state=state, updated_unix=time.time(), **values))


def job_parts(job):
    values = job.split('_')
    if values[0] == 'yolo':
        _, size, budget, seed = values
        return 'yolo', size, int(budget), int(seed[1:])
    size, budget, seed = values
    return 'songmae', size, int(budget), int(seed[1:])


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text()); verify(plan)
        return plan
    if not COMPLETION.exists():
        raise ValueError('annotation clients have not finished draining')
    completed = json.loads(COMPLETION.read_text())
    queue = json.loads((QUEUE / 'manifest.json').read_text())
    queue_sha = digest(QUEUE / 'manifest.json')
    if completed['state'] != 'paused_at_budget' or completed['manifest_sha256'] != queue_sha:
        raise ValueError('unverified annotation completion')
    names = set(completed['windows'])
    windows = [w for w in queue['windows'] if w['name'] in names]
    if len(windows) < 5500 or len({w['tile'][0] for w in windows}) != len(names):
        raise ValueError('insufficient or duplicate source recordings')
    old = json.loads((older.OUT / 'manifest.json').read_text())
    original = json.loads((trainer.OUT / 'manifest.json').read_text())
    original_val = set(original['datasets']['validation']['recording_ids'])
    previous_val = set(old['datasets']['validation']['recording_ids'])
    # Keep the original 800s validation, add 1,700s from the prior 2,875s pool.
    val = [w for w in windows if w['tile'][0] in original_val]
    if len(val) != 160 or not original_val <= previous_val:
        raise ValueError('original validation is missing')
    val += older.diverse_order([w for w in windows if w['tile'][0] in previous_val - original_val])[:340]
    val_ids = {w['tile'][0] for w in val}
    train = older.diverse_order([w for w in windows if w['tile'][0] not in val_ids])[:5000]
    if len(val) != 500 or len(train) != 5000:
        raise ValueError('wrong exact train/validation budget')
    selected = [*train, *val]
    selection = json.loads((QUEUE / 'selection.json').read_text())
    excluded = set().union(*(set(v.get('xc_ids', [])) for v in selection['exclusions'].values()))
    protected = {str(p): digest(p) for p in [COMPLETION, QUEUE/'manifest.json', QUEUE/'selection.json',
        older.OUT/'manifest.json', trainer.OUT/'manifest.json', trainer.original.CONTROL,
        ROOT/'results/qwen_teacher_powdermill/leave_one_recording_out_2026-09-14/self_review_1.json']}
    rows = {}
    for w in selected:
        name, shard, start, end, left, right = w['tile']
        if name in excluded or right-left != 1000 or not 0 <= left < right <= end-start:
            raise ValueError('excluded recording or invalid window: ' + name)
        for stage in queue['conditions']:
            if not validate(QUEUE, w, queue_sha, stage):
                raise ValueError('incomplete teacher stage: ' + w['name'])
            annotation_path = QUEUE/'annotations'/stage/f'{w["name"]}.json'
            protected[str(annotation_path)] = digest(annotation_path)
        annotation = json.loads(annotation_path.read_text())
        if annotation['events'] != trainer.teacher.parsed_events(annotation['raw_annotation'], w):
            raise ValueError('changed teacher coordinate conversion')
        raw = load_spec_slice(Path(queue['spec_dir'])/'shards'/shard, start+left, start+right)
        if hashlib.sha256(raw.tobytes()).hexdigest() != w['spectrogram_sha256']:
            raise ValueError('changed spectrogram: ' + name)
        rows[w['name']] = dict(status='ok', recording=name, source=dict(shard=shard, start=start, end=end),
            tile=dict(start_timebin=left, end_timebin=right, ownership_start_timebin=left, ownership_end_timebin=right),
            events=annotation['events'], annotation=str(annotation_path), annotation_sha256=digest(annotation_path),
            manifest_sha256=queue_sha, worker_role=annotation['worker_role'])
    configs = {size: AutoConfig.from_pretrained(m['backbone'], revision=m['revision'],
        trust_remote_code=True, local_files_only=True) for size, m in old['models'].items()}
    config = configs['large']
    for other in configs.values():
        if any(getattr(other, k) != getattr(config, k) for k in
               ['audio_mean', 'audio_std', 'mels', 'num_timebins', 'patch_height', 'patch_width']):
            raise ValueError('backbone input grids differ')
    LABELS.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for key, subset in [('validation', val), *[(str(n), train[:n//5]) for n in BUDGETS]]:
        label_path = LABELS/f'{key}.jsonl'
        text = ''.join(json.dumps(rows[w['name']], separators=(',', ':'))+'\n' for w in subset)
        if label_path.exists() and label_path.read_text() != text:
            raise ValueError('frozen labels changed')
        if not label_path.exists():
            pending = label_path.with_suffix('.pending.jsonl'); pending.write_text(text); pending.replace(label_path)
        data = PixelWindows(read_rows(label_path, 0), ROOT/'data/xcl/shards', config)
        inputs, masks = hashlib.sha256(), hashlib.sha256()
        for spec, target, valid in data:
            if valid != 1000 or not torch.isfinite(spec).all() or not torch.all((target == 0) | (target == 1)):
                raise ValueError('invalid training tensors')
            inputs.update(spec.numpy().tobytes()); masks.update(target.numpy().tobytes())
        sites = {w['coordinate_site_0_01_degree'] for w in subset}
        datasets[key] = dict(path=str(label_path), sha256=digest(label_path), seconds=len(subset)*5,
            windows=len(subset), recording_ids=sorted(w['tile'][0] for w in subset),
            events=sum(len(rows[w['name']]['events']) for w in subset),
            focal_species=len({w['metadata']['ebird_code'] for w in subset}),
            coordinate_sites=len(sites-{'unknown'}), unknown_coordinate_windows=sum(w['coordinate_site_0_01_degree']=='unknown' for w in subset),
            preflight_input_sha256=inputs.hexdigest(), preflight_mask_sha256=masks.hexdigest(), supervision=data.supervision_counts())
        protected[str(label_path)] = digest(label_path)
    for item in old['external'].values():
        if digest(item['reference_report']) != item['reference_report_sha256']:
            raise ValueError('external reference report changed')
        protected[item['reference_report']] = item['reference_report_sha256']
    for variant in ['yolo11n', 'yolo11l']:
        p = older.model_path(variant, ROOT/'models'); protected[str(p)] = digest(p)
    selected_names = {w['name'] for w in selected}
    train_ids = {w['tile'][0] for w in train}
    if train_ids & val_ids or not train_ids.isdisjoint(excluded):
        raise ValueError('recording leakage')
    # Pin current dependencies without changing any earlier study's frozen files.
    codes = set(old['code_sha256']) | {str(p) for p in ROOT.glob('scripts/paper_25k_*.py')}
    codes.add(str(ROOT/'scripts/paper_suite_evaluate.py'))
    plan = dict(run=RUN, seeds=SEEDS, epochs=10, budgets=BUDGETS, models=old['models'],
        training_config=old['training_config'], datasets=datasets, external=old['external'], powdermill=old['powdermill'],
        calibration=old['calibration'], songmae_jobs=SONGMAE_JOBS, yolo_jobs=YOLO_JOBS, external_jobs=EXTERNAL_JOBS,
        annotation=dict(completion=str(COMPLETION), completed_seconds=completed['completed_seconds'],
            selected_seconds=27500, unused_windows=sorted(names-selected_names),
            protocol='unchanged reasoning + axes + one numbered self-review'),
        split=dict(seed=0, rule='Fixed 500-recording validation, retaining original 160 and selecting 340 from previous validation; nested species/geography-balanced training.',
            previous_validation_now_training=sorted(train_ids & previous_val),
            previous_validation_unused=sorted(previous_val-train_ids-val_ids),
            note='New experimental split; some prior 15k-study validation recordings may now train. No current validation or evaluation recording enters current training. Fresh pretrained initialization, not earlier fine-tuned checkpoints.'),
        training='Full encoder including CNN/positions + bare linear; hard BCE; ten epochs, best XC validation BCE; seeds 0,1,2 vary initialization/order only.',
        yolo='Both released BirdBox sizes fine-tuned at every budget, ten epochs, best XC validation box mAP50-95, seeds 0,1,2. Native YOLO inputs and optimization; no SongMAE smoothing.',
        reporting='Mean and sample SD across all three seeds; each selected checkpoint evaluated independently. No best-seed selection or ensemble.',
        caveat='Powdermill is development data. Equal epochs are not equal compute/updates across budgets or architectures. Metadata species need not vocalize within every window. Source-ID exclusions do not prove absence of reuploads or pretraining overlap.',
        protected_files=protected, code_sha256={p:digest(p) for p in sorted(codes)})
    verify(plan); frozen(path, plan)
    print('Frozen budgets:', {k:v['seconds'] for k,v in datasets.items()}, flush=True)
    return plan


def train_songmae(job):
    master = prepare(); family, size, budget, seed = job_parts(job)
    if family != 'songmae' or job not in SONGMAE_JOBS:
        raise ValueError('unknown SongMAE job')
    out, artifacts = OUT/'runs'/job, ART/'songmae'/job
    out.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    lock = (out/'driver.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = dict(master, seed=seed, datasets=dict(train=master['datasets'][str(budget)], validation=master['datasets']['validation']))
    frozen(out/'manifest.json', plan); trainer.OUT = out
    checkpoint = artifacts/'model.pt'
    if not checkpoint.exists() and torch.cuda.mem_get_info(0)[0] < 20*1024**3:
        raise ValueError('GPU occupied')
    trainer.train(plan, size, out, artifacts, checkpoint)
    if not (out/'loro.json').exists():
        # The earlier helper hardcodes zero in its report label, not in training.
        # Override that metadata at its aggregation boundary for all new seeds.
        trainer.cross_calibrate = lambda spec, report, rows: cross_calibrate(
            dict(spec, id=job, seed=seed), report, rows)
        trainer.evaluate(plan, size, out, artifacts, checkpoint)
    result = json.loads((out/'loro.json').read_text())
    if result['seed'] != seed or result['summary']['segments'] != 77:
        raise ValueError('evaluation seed/coverage differs')
    verify(master)
    atomic(out/'status.json', dict(state='complete', job=job, seed=seed, training_seconds=budget, updated_unix=time.time()))


def invoke(script, args, gpu, name, python=None):
    older.invoke(script, args, gpu, OUT/'logs'/f'{name}.log', python)


def lanes(jobs, function):
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(lambda i=gpu: [function(job, i) for job in jobs[i::2]]) for gpu in [0,1]]
        for future in futures:
            future.result()


def evaluate_external(job, gpu):
    python = ROOT/'.venv-birdcode/bin/python' if job == 'birdcode' else None
    for dataset in ['powdermill', 'xcsl', 'nips4bplus', 'wabad', 'hawaii']:
        if job == 'birdcode' and dataset in ['wabad', 'hawaii']:
            continue
        invoke('paper_25k_evaluate.py', ['--model', job, '--dataset', dataset], gpu, f'{job}_{dataset}', python)


def driver():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'driver.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        status('waiting_for_annotation_drain')
        while not COMPLETION.exists():
            state = subprocess.check_output(['systemctl','--user','show','birdsong-qwen-budget-27500-20260915.service','-p','ActiveState','--value'], text=True).strip()
            if state not in ['active', 'activating']:
                raise ValueError('annotation watcher stopped without completion')
            time.sleep(10)
        subprocess.run(['systemctl','--user','disable','--now','birdsong-qwen-budget-27500-20260915.service'], check=True)
        for role in ['twins', 'server', 'v100']:
            if subprocess.check_output(['systemctl','--user','show',f'birdsong-qwen-xc50k-{role}-20260914.service','-p','MainPID','--value'], text=True).strip() != '0':
                raise ValueError('annotation has not paused: ' + role)
        if shutil.disk_usage(ART).free < 150*1024**3 or shutil.disk_usage(ROOT).free < 512*1024**2:
            raise ValueError('insufficient output disk space')
        if any(torch.cuda.mem_get_info(i)[0] < 20*1024**3 for i in [0,1]):
            raise ValueError('Twins GPU occupied by another job')
        status('preparing_frozen_split'); prepare()
        if (OUT/'complete.json').exists():
            status('complete'); return
        for stage, jobs in [('large_scaling', SONGMAE_JOBS[:15]), ('backbone_sizes', SONGMAE_JOBS[15:])]:
            status(stage)
            lanes(jobs, lambda job,gpu: invoke('paper_25k_suite.py', ['--train',job],gpu,job))
        invoke('paper_25k_report.py', ['--phase','songmae'],0,'report_songmae')
        status('songmae_external')
        lanes(EXTERNAL_JOBS[:3], evaluate_external)
        invoke('paper_25k_report.py', ['--phase','partial'],0,'report_songmae_external')
        status('yolo_training_and_powdermill')
        def yolo_job(job,gpu):
            invoke('paper_25k_yolo.py',['--job',job],gpu,job)
            invoke('paper_25k_evaluate.py',['--model',job,'--dataset','powdermill'],gpu,f'{job}_powdermill')
        lanes(YOLO_JOBS,yolo_job)
        status('yolo_external'); lanes(EXTERNAL_JOBS[3:],evaluate_external)
        status('released_competitors'); lanes(RELEASED,evaluate_external)
        invoke('paper_25k_report.py',['--phase','final'],0,'report_final')
        verify(prepare())
        atomic(OUT/'complete.json',dict(state='complete',completed_unix=time.time(),manifest_sha256=digest(OUT/'manifest.json')))
        status('complete')
    except BaseException as error:
        status('failed',error=repr(error)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--train', choices=SONGMAE_JOBS)
    args = parser.parse_args(); os.chdir(ROOT)
    if args.prepare_only:
        prepare()
    elif args.train:
        train_songmae(args.train)
    else:
        driver()
