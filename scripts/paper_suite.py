#!/usr/bin/env python3
"""Frozen 15k XC suite: nested budgets, backbone sizes, native competitors, paper outputs."""
import argparse
import collections
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

import run_full_encoder_backbones_snapshot as previous
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.data import PixelWindows, load_spec_slice, read_rows
from birdsong_detect_distill.birdbox import model_path
from prepare_detector_study import verify
from run_qwen_xc_50k import validate

ROOT = Path(__file__).resolve().parents[1]
RUN = 'paper_15000train_2026-09-15'
OUT = ROOT / 'results' / RUN
ART = previous.ARTIFACTS.parent / RUN
LABELS = ROOT / 'data/annotations/xcl' / RUN
QUEUE = previous.QUEUE
RAW = previous.original.RAW
OLD_EXTERNAL = ROOT / 'results/current_external_2026-09-09/manifest.json'
BUDGETS = [100, 1000, 10000, 15000]
SONGMAE_JOBS = [f'large_{n}' for n in BUDGETS] + ['micro_15000', 'base_15000']
MODELS = ['songmae', 'released_n', 'released_l', 'teacher_n', 'teacher_l', 'birdcode']


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    write_json(temporary, value)
    temporary.replace(path)


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen file changed: {path}')
    else:
        atomic(path, value)


def status(state, **values):
    atomic(OUT / 'status.json', dict(state=state, updated_unix=time.time(), **values))


def snapshot():
    path = OUT / 'snapshot.json'
    if path.exists():
        return json.loads(path.read_text())
    plan = json.loads((QUEUE / 'manifest.json').read_text())
    sha = digest(QUEUE / 'manifest.json')
    names = {p.stem for p in (QUEUE / 'annotations/self_review_1').glob('*.json')}
    selected = [w for w in plan['windows'] if w['name'] in names]
    for w in selected:
        for condition in plan['conditions']:
            if not validate(QUEUE, w, sha, condition):
                raise ValueError('incomplete selected annotation')
    value = dict(created_unix=time.time(), manifest_sha256=sha,
        names=[w['name'] for w in selected], seconds=len(selected)*5,
        policy='Completed reasoning and one self-review only; later V100/Twins completions excluded from this experiment.')
    frozen(path, value)
    print(json.dumps({k:v for k,v in value.items() if k!='names'}), flush=True)
    return value


def diverse_order(windows):
    rng = random.Random(0)
    buckets = collections.defaultdict(list)
    for w in windows:
        buckets[w['metadata']['ebird_code']].append(w)
    for values in buckets.values():
        rng.shuffle(values)
    ordered, cells = [], collections.Counter()
    while buckets:
        species = list(buckets)
        rng.shuffle(species)
        for code in species:
            choices = buckets[code]
            w = min(choices, key=lambda w:cells[w['geographic_cell_1_degree']])
            choices.remove(w)
            ordered.append(w)
            cells[w['geographic_cell_1_degree']] += 1
            if not choices:
                del buckets[code]
    return ordered


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        verify(plan)
        return plan
    snap = snapshot()
    queue = json.loads((QUEUE / 'manifest.json').read_text())
    if digest(QUEUE / 'manifest.json') != snap['manifest_sha256']:
        raise ValueError('annotation manifest changed')
    names = set(snap['names'])
    windows = [w for w in queue['windows'] if w['name'] in names]
    if len({w['tile'][0] for w in windows}) != len(windows):
        raise ValueError('expected one selected window per source recording')
    older = json.loads((previous.OUT / 'manifest.json').read_text())
    old_val = set(older['datasets']['validation']['recording_ids'])
    if not old_val <= {w['tile'][0] for w in windows}:
        raise ValueError('original validation recordings missing from snapshot')
    order = diverse_order([w for w in windows if w['tile'][0] not in old_val])
    if len(order) < 3000:
        raise ValueError('insufficient recording-disjoint data for 15,000 training seconds')
    train = order[:3000]
    train_ids = {w['tile'][0] for w in train}
    val = [w for w in windows if w['tile'][0] not in train_ids]
    selection = json.loads((QUEUE / 'selection.json').read_text())
    excluded = set().union(*(set(v.get('xc_ids', [])) for v in selection['exclusions'].values()))
    protected = {str(p):digest(p) for p in [OUT/'snapshot.json', QUEUE/'manifest.json', QUEUE/'selection.json', OLD_EXTERNAL,
        previous.OUT/'manifest.json',previous.original.CONTROL,
        ROOT/'results/qwen_teacher_powdermill/leave_one_recording_out_2026-09-14/self_review_1.json']}
    rows = {}
    for w in windows:
        name, shard, start, end, left, right = w['tile']
        if name in excluded or right-left != 1000 or not 0 <= left < right <= end-start:
            raise ValueError('excluded recording or invalid interval')
        for stage in queue['conditions']:
            if not validate(QUEUE, w, snap['manifest_sha256'], stage):
                raise ValueError('changed teacher annotation')
            p = QUEUE/'annotations'/stage/f'{w["name"]}.json'
            protected[str(p)] = digest(p)
        annotation = json.loads(p.read_text())
        if annotation['events'] != previous.teacher.parsed_events(annotation['raw_annotation'], w):
            raise ValueError('teacher coordinate conversion differs')
        raw = load_spec_slice(Path(queue['spec_dir'])/'shards'/shard, start+left, start+right)
        if hashlib.sha256(raw.tobytes()).hexdigest() != w['spectrogram_sha256']:
            raise ValueError('teacher/student spectrogram mismatch')
        rows[w['name']] = dict(status='ok', recording=name, source=dict(shard=shard,start=start,end=end),
            tile=dict(start_timebin=left,end_timebin=right,ownership_start_timebin=left,ownership_end_timebin=right),
            events=annotation['events'], annotation=str(p), annotation_sha256=digest(p),
            manifest_sha256=snap['manifest_sha256'], worker_role=annotation['worker_role'])
    config = AutoConfig.from_pretrained('georgeven/songmae-large-32x1',
        revision=previous.scorer.REVISIONS['large'], trust_remote_code=True, local_files_only=True)
    for size,revision in previous.scorer.REVISIONS.items():
        other = AutoConfig.from_pretrained(f'georgeven/songmae-{size}-32x1',revision=revision,
            trust_remote_code=True,local_files_only=True)
        if any(getattr(other,k)!=getattr(config,k) for k in ['audio_mean','audio_std','mels','num_timebins','patch_height','patch_width']):
            raise ValueError('backbone input grids or normalization differ')
    datasets = {}
    for key, subset in [('validation', val), *[(str(n), train[:n//5]) for n in BUDGETS]]:
        target = LABELS/f'{key}.jsonl'
        content = ''.join(json.dumps(rows[w['name']],separators=(',',':'))+'\n' for w in subset)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_text() != content:
            raise ValueError('existing split differs')
        if not target.exists():
            pending = target.with_suffix('.pending.jsonl')
            pending.write_text(content)
            pending.replace(target)
        data = PixelWindows(read_rows(target,0),ROOT/'data/xcl/shards',config)
        inputs, masks = hashlib.sha256(), hashlib.sha256()
        for spec, mask, valid in data:
            if valid!=1000 or not torch.isfinite(spec).all() or not torch.all((mask==0)|(mask==1)):
                raise ValueError('invalid training tensor')
            inputs.update(spec.numpy().tobytes())
            masks.update(mask.numpy().tobytes())
        datasets[key] = dict(path=str(target),sha256=digest(target),seconds=len(data)*5,windows=len(data),
            recording_ids=sorted(w['tile'][0] for w in subset),preflight_input_sha256=inputs.hexdigest(),
            preflight_mask_sha256=masks.hexdigest(),supervision=data.supervision_counts(),
            events=sum(len(rows[w['name']]['events']) for w in subset),
            species=len({w['metadata']['ebird_code'] for w in subset}),
            coordinate_sites=len({w['coordinate_site_0_01_degree'] for w in subset}))
        protected[str(target)] = digest(target)
        del data
    external = json.loads(OLD_EXTERNAL.read_text())
    for item in external['external'].values():
        protected[item['reference_report']] = item['reference_report_sha256']
    for variant in ['yolo11n','yolo11l']:
        p = model_path(variant,ROOT/'models')
        protected[str(p)] = digest(p)
    codes = set(older['code_sha256']) | {str(p) for p in ROOT.glob('scripts/paper_suite*.py')}
    codes |= {str(ROOT/'scripts'/p) for p in ['prepare_yolo.py','evaluate_current_models.py',
        'evaluate_baselines.py','merge_external_shards.py','run_full_encoder_backbones_snapshot.py']}
    codes |= {str(p) for p in (ROOT/'src/birdsong_detect_distill').glob('*.py')}
    plan = dict(run=RUN,seed=0,epochs=5,datasets=datasets,training_config=older['training_config'],
        models=older['models'],snapshot=snap,external=external['external'],powdermill=external['powdermill'],
        teacher_condition='reasoning + one numbered self-review, coordinate axes',
        split='15,000 s species-round-robin training; all remaining snapshot recordings validation, including previous 800 s validation. Nested budgets, seed 0; no recording overlap.',
        training='Full encoder including CNN and positional embeddings + bare linear; hard BCE; five epochs; best XC validation BCE; same recipe as 2026-09-14.',
        calibration='Separate per-model full-band/common-band/temporal mean-IoU thresholds on Powdermill Recordings 2–4; frozen for external evaluation. Full-Powdermill figure uses leave-one-original-recording-out calibration.',
        yolo='Released human-trained YOLO11n and YOLO11l plus 50-epoch fine-tuning of each on identical 15k XC labels; best XC validation box mAP; native released inputs preserved.',
        caveat='Single seed. Powdermill is development data. Different pretraining, representations and optimization budgets preclude an architecture-only causal claim. Known-ID exclusions do not certify all pretraining provenance.',
        protected_files=protected,code_sha256={p:digest(p) for p in sorted(codes)})
    verify(plan)
    frozen(path,plan)
    print('Frozen:',{k:(v['seconds'],v['species']) for k,v in datasets.items()},flush=True)
    return plan


def train_songmae(job):
    master = prepare()
    size, budget = job.split('_')
    out, artifacts = OUT/'runs'/job, ART/'songmae'/job
    out.mkdir(parents=True,exist_ok=True)
    artifacts.mkdir(parents=True,exist_ok=True)
    plan = dict(master, datasets=dict(train=master['datasets'][budget],validation=master['datasets']['validation']))
    frozen(out/'manifest.json',plan)
    previous.OUT = out
    if (out/'loro.json').exists():
        previous.train(plan,size,out,artifacts,artifacts/'model.pt')
        return
    if torch.cuda.mem_get_info(0)[0] < 20*1024**3:
        raise ValueError('GPU occupied; preserve other jobs')
    previous.train(plan,size,out,artifacts,artifacts/'model.pt')
    previous.evaluate(plan,size,out,artifacts,artifacts/'model.pt')
    verify(master)
    previous.progress(out,'complete',size=size,budget_seconds=int(budget))


def invoke(script, arguments, gpu, log, python=None):
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1',
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
    log.parent.mkdir(parents=True,exist_ok=True)
    command = [str(python or ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts'/script),*map(str,arguments)]
    print('Running:', ' '.join(command),flush=True)
    with log.open('a') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def lanes(jobs, function):
    def lane(gpu):
        for job in jobs[gpu::2]:
            function(job,gpu)
    with ThreadPoolExecutor(2) as pool:
        for result in [pool.submit(lane,gpu) for gpu in range(2)]:
            result.result()


def driver():
    OUT.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'driver.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        plan = prepare()
        if (OUT/'complete.json').exists():
            return
        status('draining_twins_annotations')
        unit = 'birdsong-qwen-xc50k-twins-20260914.service'
        # The client saves its current calls before exiting; never stop the model first.
        subprocess.run(['systemctl','--user','stop','--no-block',unit],check=True)
        deadline = time.monotonic()+7600
        while subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True).strip()!='0':
            if time.monotonic()>deadline:
                raise RuntimeError('Twins did not drain; no GPU work launched')
            time.sleep(5)
        subprocess.run(['systemctl','--user','stop','birdsong-qwen-xc50k-server-20260914.service'],check=True)
        ART.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(ART).free<100*1024**3 or shutil.disk_usage(ROOT).free<1024**3:
            raise ValueError('insufficient disk headroom')
        if any(torch.cuda.mem_get_info(i)[0]<20*1024**3 for i in [0,1]):
            raise ValueError('GPU still occupied')
        status('songmae_label_budget',validation_seconds=plan['datasets']['validation']['seconds'])
        run = lambda job,gpu:invoke('paper_suite.py',['--train',job],gpu,OUT/'logs'/f'{job}.log')
        lanes(['large_15000','large_10000','large_1000','large_100'],run)
        status('songmae_backbone_sizes')
        lanes(['micro_15000','base_15000'],run)
        invoke('paper_suite_report.py',['--development-only'],0,OUT/'logs/development_report.log')
        status('yolo_training')
        lanes(['n','l'],lambda v,gpu:invoke('paper_suite_yolo.py',['--variant',v],gpu,OUT/'logs'/f'yolo_{v}_training.log'))
        status('external_evaluation')
        def evaluate_model(model,gpu):
            python = ROOT/'.venv-birdcode/bin/python' if model=='birdcode' else None
            for dataset in ['powdermill','xcsl','nips4bplus','wabad','hawaii']:
                if model=='birdcode' and dataset in ['wabad','hawaii']:
                    continue
                invoke('paper_suite_evaluate.py',['--model',model,'--dataset',dataset],gpu,
                    OUT/'logs'/f'{model}_{dataset}.log',python)
        lanes(MODELS,evaluate_model)
        invoke('paper_suite_report.py',[],0,OUT/'logs/final_report.log')
        verify(plan)
        atomic(OUT/'complete.json',dict(state='complete',completed_unix=time.time(),manifest_sha256=digest(OUT/'manifest.json')))
        status('complete',results=str(OUT/'tables.md'),figures=str(OUT/'figures'))
    except Exception as error:
        status('failed',error=repr(error))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--train',choices=SONGMAE_JOBS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.snapshot:
        snapshot()
    elif args.prepare:
        prepare()
    elif args.train:
        train_songmae(args.train)
    else:
        driver()
