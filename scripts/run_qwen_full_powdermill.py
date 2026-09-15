#!/usr/bin/env python3
"""Full Powdermill extension of the frozen 250-window, five-condition pilot."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import requests

import qwen_prompt_study as study
import run_qwen_prompt_study as engine

ROOT=study.ROOT
PILOT=study.OUT
OUT=ROOT/'results/qwen_teacher_powdermill/prompt_full_2026-09-11'
SERVER='birdsong-qwen-full-server-20260911.service'
CONDITIONS=['reasoning','self_review_1','self_review_2','direct_plain','direct_axes']
PARENT={'self_review_1':'reasoning','self_review_2':'self_review_1'}
BASE_ANNOTATE=engine.annotate


def prepare():
    pilot=study.prepare()  # Validate the unchanged pilot and its review amendment.
    path=OUT/'manifest.json'
    if path.exists():
        plan=json.loads(path.read_text())
        for name,sha in plan['protected'].items():
            if study.digest(name)!=sha: raise ValueError('frozen input changed: '+name)
        return plan
    selected={w['name']:w for w in pilot['windows']}
    windows=[]; extra=0
    for tile in sorted(study.split_tiles(study.read_tiles(study.SPEC/'recordings.jsonl'),1000)):
        if tile[-1]-tile[-2]!=1000: continue  # Centered-STFT endpoint, not extra audio.
        name=f'{tile[0]}_{tile[-2]}_{tile[-1]}'
        if name in selected: seed=selected[name]['seed']
        else: seed=17+31*(250+extra); extra+=1
        windows.append(dict(name=name,tile=list(tile),seed=seed,
            partition='evaluation' if tile[0].startswith('Recording_1_') else 'calibration',
            reused_pilot_window=name in selected))
    assert len(windows)==4620 and extra==4370 and len({w['seed'] for w in windows})==4620
    protected={str(PILOT/'manifest.json'):study.digest(PILOT/'manifest.json'),
        str(PILOT/'review_amendment.json'):study.digest(PILOT/'review_amendment.json')}
    # Preserve source arrays, affine sidecars, references, code, fonts, and reused results.
    paths=list(map(Path,pilot['protected']))+list((study.SPEC/'shards').glob('*'))
    paths += [Path(__file__),ROOT/'src/birdsong_detect_distill/data.py',PILOT/'axis_decision.json',
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
        Path('/home/george-vengrovski/Documents/SongMAE/shell/qwen38_27b_server.sh')]
    paths += list((PILOT/'annotations').glob('*/*.json'))
    for p in paths:
        if p.is_file(): protected[str(p.resolve())]=study.digest(p)
    plan={**pilot,'windows':windows,'protected':protected,'conditions':CONDITIONS,
        'sampling':'Exhaustive 77 x 300-second segments, 60 consecutive nonoverlapping windows each; 23,100 seconds. No endpoint-only pseudo-windows.',
        'selected_image_format':'axes','axis_selection':'Frozen from the original pilot calibration decision; not reselected on full Powdermill.',
        'coverage':dict(evaluation_windows=2160,calibration_windows=2460,seconds=23100),
        'seed_policy':'Preserve all 250 pilot seeds; remaining windows receive unique seeds 17+31*(250+index) in sorted recording/time order.',
        'execution':'Reasoning, first review, second review, plain direct control, axes direct control. Each stage saved and scored before advancing.',
        'image_storage':'Render identical PNG bytes on demand from frozen arrays; remove only per-call temporary copies. Input image hashes retained.',
        'pilot_manifest_sha256':study.digest(PILOT/'manifest.json'),
        'metrics':'Segment-macro pixel AP/IoU, pooled precision/recall. Calibrate IoU independently on full Recordings_2–4, report full Recording_1. Also save calibration summaries.',
        'server_service':SERVER,'caveat':'Development-set extension, not an independent test; includes the 250 pilot windows. No student training.'}
    study.write(path,plan)
    return plan


def configure():
    study.OUT=engine.OUT=OUT
    engine.SERVER=SERVER
    engine.annotate=annotate


def import_pilot(plan):
    manifest_hash=study.digest(OUT/'manifest.json')
    for condition in CONDITIONS:
        for window in plan['windows']:
            if not window['reused_pilot_window']: continue
            source=PILOT/'annotations'/condition/f'{window["name"]}.json'
            target=OUT/'annotations'/condition/source.name
            if target.exists():
                saved=json.loads(target.read_text())
                assert saved['manifest_sha256']==manifest_hash and saved['reused_from_sha256']==study.digest(source)
                continue
            saved=json.loads(source.read_text())
            assert saved['manifest_sha256']==plan['pilot_manifest_sha256'] and saved['mode_verified']
            assert saved.get('review_amendment_sha256')==study.review_revision(plan,condition)
            parent=PARENT.get(condition)
            parent_hash=study.digest(OUT/'annotations'/parent/source.name) if parent else None
            picture=PILOT/'images'/saved['mode']/f'{window["name"]}.png'
            saved.update(manifest_sha256=manifest_hash,parent_annotation_sha256=parent_hash,
                reused_from=str(source),reused_from_sha256=study.digest(source),input_image_sha256=study.digest(picture))
            study.write(target,saved)


@contextmanager
def rendered(window,mode):
    scratch=OUT/'scratch'; scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='window-',dir=scratch) as folder:
        spec,_,_=study.context(study.SPEC,window['tile'],*window['tile'][-2:])
        picture=Path(folder)/'input.png'
        study.decorate(study.image(spec),mode=='axes').save(picture)
        yield {**window,'images':{mode:str(picture)}},picture


def annotate(window,condition,mode,plan):
    target=OUT/'annotations'/condition/f'{window["name"]}.json'
    if target.exists():
        saved=json.loads(target.read_text()); parent=PARENT.get(condition)
        assert saved['manifest_sha256']==study.digest(OUT/'manifest.json') and saved['mode']==mode and saved['mode_verified']
        assert saved.get('review_amendment_sha256')==study.review_revision(plan,condition)
        assert saved['parent_annotation_sha256']==(study.digest(OUT/'annotations'/parent/target.name) if parent else None)
        return
    if engine.STOP.is_set(): return
    with rendered(window,mode) as (ready,picture):
        BASE_ANNOTATE(ready,condition,mode,plan)
        if target.exists():
            saved=json.loads(target.read_text()); saved['input_image_sha256']=study.digest(picture)
            study.write(target,saved)


def report(plan):
    lines=['# Full Powdermill: simplified Qwen prompt','',
        '| Configuration | Pixel AP | 2D IoU | Pixel precision | Pixel recall | Threshold |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for condition in ['direct_plain','direct_axes','reasoning','self_review_1','self_review_2']:
        path=OUT/'scores'/f'{condition}.json'
        if not path.exists(): continue
        r=json.loads(path.read_text())['evaluation']
        lines.append(f'| {condition} | {r["ap"]:.4f} | {r["iou"]:.4f} | {r["pooled_precision"]:.4f} | {r["pooled_recall"]:.4f} | {r["threshold"]:.2f} |')
    lines += ['','All 4,620 windows / 23,100 seconds are annotated for each completed condition.',
        'Report: full Recording_1 (2,160 windows, 10,800 s). Calibrate: full Recordings_2–4 (2,460 windows, 12,300 s).',
        'Axes fixed from the pilot. Same prompts, numbered reviews, seeds for reused windows, and scoring method.',
        '250 pilot outputs reused per condition with original provenance. No historical results overwritten.',
        'Timing in scores/ covers newly executed calls only; metrics include the reused pilot windows.',
        'Development-set results, not an independent test or student-model evaluation.']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')


def run(plan):
    status=OUT/'status.json'
    if status.exists() and json.loads(status.read_text()).get('state')=='complete': return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPUs occupied; refusing to interfere')
    try:
        if requests.get(engine.URL+'/health',timeout=2).ok: raise ValueError('port 8080 is already occupied')
    except requests.ConnectionError: pass
    subprocess.run(['systemctl','--user','start',SERVER],check=True)
    try:
        deadline=time.monotonic()+180
        while True:
            if engine.STOP.is_set(): raise InterruptedError('paused during startup')
            try: ready=requests.get(engine.URL+'/health',timeout=2).ok
            except requests.RequestException: ready=False
            if ready: break
            if time.monotonic()>deadline: raise RuntimeError('server startup timed out')
            time.sleep(2)
        props=requests.get(engine.URL+'/props',timeout=10).json()
        assert props['total_slots']==16 and props['default_generation_settings']['n_ctx']==16384
        study.write(OUT/f'server_props-{time.time_ns()}.json',props)
        for condition in CONDITIONS:
            mode='plain' if condition=='direct_plain' else 'axes'
            with rendered(plan['windows'][0],mode) as (window,_):
                engine.preflight({**plan,'windows':[window]},condition,mode)
            engine.stage([(w,condition,mode) for w in plan['windows']],plan,condition)
            study.verify(plan); study.score(plan,condition); report(plan)
            print('Scored complete full-Powdermill condition: '+condition,flush=True)
        study.write(status,dict(state='complete',conditions=CONDITIONS,windows=4620,accepted_outputs=23100,
            reused_pilot_outputs=1250,new_outputs=21850,selected_image_format='axes'))
    finally:
        subprocess.run(['systemctl','--user','stop',SERVER],check=True)


if __name__=='__main__':
    os.chdir(ROOT)
    for sig in [signal.SIGINT,signal.SIGTERM]: signal.signal(sig,lambda *_:engine.STOP.set())
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare',action='store_true'); args=parser.parse_args()
    plan=prepare(); configure(); import_pilot(plan)
    if args.prepare: print('Frozen all 4,620 windows and imported 1,250 pilot outputs.',flush=True)
    else:
        try: run(plan)
        except BaseException as error:
            study.write(OUT/'status.json',dict(state='paused' if isinstance(error,InterruptedError) else 'failed',error=str(error)))
            raise
