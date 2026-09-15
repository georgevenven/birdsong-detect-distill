#!/usr/bin/env python3
"""Verified no-reasoning pilot against historical direct labels on 100 matched windows."""
import argparse
import base64
import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

from birdsong_detect_distill.benchmark_data import digest, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, THRESHOLDS, area, calibrate, summarize
from birdsong_detect_distill.qwen import SYSTEM, LABELS, canonical_events, context, image, map_final, ownership_x, schema
from evaluate_stage_pipeline import teacher_probability
from evaluate_songmae_smoothing import truth_and_coverage

OUT = Path('results/qwen_teacher_powdermill/no_reason_100_2026-09-11')
MASTER = Path('data/annotations/powdermill/qwen_ablation/passes.jsonl')
SPEC = Path('data/powdermill/qwen')
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
SERVER = 'birdsong-qwen-pilot-server-20260911.service'
URL = 'http://127.0.0.1:8080'
SCRIPT = 'scripts/run_qwen_no_reason_pilot.py'


def verify(plan):
    for path, sha in plan['protected'].items():
        if digest(path) != sha:
            raise ValueError(f'frozen input changed: {path}')


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        verify(plan)
        return plan
    master = {}
    for row in map(json.loads, MASTER.open()):
        if row.get('status') == 'ok':
            master[row['recording'], row['owner_start'], row['owner_end']] = row
    excluded = {('Recording_1_Segment_23',2000,3000), ('Recording_1_Segment_23',3000,4000),
        ('Recording_1_Segment_25',48000,49000), ('Recording_4_Segment_23',0,1000)}
    pools = defaultdict(list)
    for source in sorted(MASTER.parent.joinpath('stage_checkpoints').glob('*/*.json')):
        state = json.loads(source.read_text())
        tile = state['tile']
        key = tile[0], tile[-2], tile[-1]
        old = master.get(key)
        if key in excluded or tile[-1]-tile[-2] != 1000 or old is None:
            continue
        if state['stages'].get('direct') != old['passes']['direct']:
            continue
        pools[tile[0].split('_Segment_')[0]].append((source, state, old))
    # Source-stratified, label-blind sampling; no selection by predicted/true foreground.
    selected = []
    for group, count in [('Recording_1',60), ('Recording_2',14), ('Recording_3',13), ('Recording_4',13)]:
        selected.extend(random.Random(17).sample(pools[group], count))
    protected = {str(MASTER):digest(MASTER), str(SPEC / 'audio_params.json'):digest(SPEC / 'audio_params.json')}
    windows = []
    for source, state, old in selected:
        tile = state['tile']
        name = f'{tile[0]}_{tile[-2]}_{tile[-1]}'
        spec, start, end = context(SPEC, tile, tile[-2], tile[-1])
        picture = OUT / 'images' / f'{name}.png'
        picture.parent.mkdir(parents=True, exist_ok=True)
        if picture.exists():
            raise ValueError('partial preparation exists without a manifest; inspect before replacing')
        image(spec).save(picture)
        protected[str(source)] = digest(source)
        protected[str(picture)] = digest(picture)
        windows.append(dict(name=name, tile=tile, seed=state['seed'], context=[start,end],
            partition='evaluation' if tile[0].startswith('Recording_1_') else 'calibration',
            image=str(picture), checkpoint=str(source), historic=old['variants']['direct']))
    for source in [SCRIPT, 'src/birdsong_detect_distill/qwen.py', 'src/birdsong_detect_distill/data.py',
            'src/birdsong_detect_distill/benchmark_data.py', 'src/birdsong_detect_distill/benchmark_metrics.py',
            'scripts/evaluate_stage_pipeline.py', 'scripts/evaluate_songmae_smoothing.py']:
        protected[source] = digest(source)
    protected[str(RAW / 'powdermill/annotation_Files.zip')] = digest(RAW / 'powdermill/annotation_Files.zip')
    plan = dict(windows=windows, protected=protected, workers=32, context_per_slot=8192,
        selection='seed-17 label-blind sample from full windows with saved original seeds; 60 Recording_1, 14/13/13 Recordings_2/3/4',
        limitations='small checkpoint-available development subset; historical direct was not verified non-reasoning',
        requested_mode=dict(reasoning_effort='none', reasoning_budget_tokens=0, enable_thinking=False, reasoning_format='deepseek'),
        max_tokens=4096, length_retry_max_tokens=6144, temperature=.5, top_p=.95, top_k=20,
        system_prompt=SYSTEM, schema=schema(),
        evaluation='segment-macro AP on continuous box scores; per-model IoU threshold calibrated on the 40 separate windows; report 60 windows',
        threshold_grid=THRESHOLDS.tolist(), teacher_smoothing=False)
    write_json(path, plan)
    return plan


def payload(window, plan, limit):
    start, end = window['context']
    own = ownership_x(window['tile'][-2], window['tile'][-1], start, end)
    instruction = f'Ownership is x={own[0]}..{own[1]} on the 0-1000 axis. Return only events whose midpoint lies inside it.'
    encoded = base64.b64encode(Path(window['image']).read_bytes()).decode()
    return dict(model='qwen3.8-27b-q8', messages=[dict(role='system',content=plan['system_prompt']),
        dict(role='user',content=[dict(type='text',text=instruction),
            dict(type='image_url',image_url={'url':'data:image/png;base64,'+encoded})])],
        temperature=plan['temperature'], top_p=plan['top_p'], top_k=plan['top_k'], seed=window['seed'],
        max_tokens=limit, reasoning_effort='none', reasoning_budget_tokens=0, reasoning_format='deepseek',
        chat_template_kwargs=dict(add_vision_id=True,enable_thinking=False),
        response_format=dict(type='json_schema',json_schema=dict(name='spectrogram_events',strict=True,schema=plan['schema'])))


def annotate(window, plan, abort):
    saved = OUT / 'annotations' / f'{window["name"]}.json'
    if saved.exists():
        result = json.loads(saved.read_text())
        if result['manifest_sha256'] != digest(OUT / 'manifest.json') or not result['no_reasoning_verified']:
            raise ValueError('cached annotation belongs to a different or unverified run')
        return result
    for attempt in range(3):
        if abort.is_set():
            raise ValueError('another request failed reasoning verification')
        request = payload(window,plan,plan['max_tokens'] if attempt==0 else plan['length_retry_max_tokens'])
        started = time.time()
        response = requests.post(URL+'/v1/chat/completions',json=request,timeout=900)
        elapsed = time.time()-started
        response.raise_for_status()
        raw = response.json()
        choice = raw['choices'][0]
        message = choice['message']
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        content = message.get('content') or ''
        reasoning_tokens = raw.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens')
        verified = not reasoning and reasoning_tokens in (None,0) and '<think>' not in content and '</think>' not in content
        audit = dict(started_unix=started, elapsed_seconds=elapsed, seed=window['seed'],
            request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
            max_tokens=request['max_tokens'], finish_reason=choice['finish_reason'],
            response_id=raw.get('id'), model=raw.get('model'), usage=raw.get('usage'), timings=raw.get('timings'),
            reasoning_characters=len(reasoning), reasoning_tokens=reasoning_tokens,
            no_reasoning_verified=verified, content=content)
        write_json(OUT / 'attempts' / f'{window["name"]}_{attempt}.json',audit)
        if not verified:
            abort.set()
            raise ValueError(f'reasoning appeared in a disabled request: {window["name"]}')
        if choice['finish_reason']=='length':
            continue
        if choice['finish_reason']!='stop':
            raise ValueError(f'unexpected finish reason: {choice["finish_reason"]}')
        result = json.loads(content)
        if set(result)!= {'events','window_quality','public_summary'} or not isinstance(result['events'],list):
            raise ValueError('invalid annotation JSON')
        for event in result['events']:
            box = event['bbox_2d']
            if (len(box)!=4 or any(type(x)!=int or not 0<=x<=1000 for x in box)
                    or event['label'] not in LABELS or not 0<=event['confidence']<=1):
                raise ValueError('invalid event coordinates, confidence or label')
        start,end = window['context']
        left,right = window['tile'][-2:]
        events = canonical_events(result['events'],start,end,start,end,left,right)
        output = dict(**{k:v for k,v in audit.items() if k!='content'}, manifest_sha256=digest(OUT / 'manifest.json'),
            raw_annotation=result, events=map_final(events,start,end,left,right,128,5))
        write_json(saved,output)
        return output
    raise ValueError(f'JSON still truncated after three attempts: {window["name"]}')


def evaluate(plan):
    inventory = {r['name']:r for r in recordings(RAW,'powdermill')}
    grouped = defaultdict(list)
    for window in plan['windows']:
        grouped[window['tile'][0]].append(window)
    scores = {model:{part:[] for part in ['calibration','evaluation']} for model in ['historical_direct','verified_no_reasoning']}
    coverage = {}
    for name, windows in sorted(grouped.items()):
        width = max(w['tile'][3]-w['tile'][2] for w in windows)
        intervals = [[w['tile'][-2]/RATE,w['tile'][-1]/RATE] for w in windows]
        truth,kept = truth_and_coverage(inventory[name],width,intervals)
        assert kept.sum()==len(windows)*1000
        coverage[name] = dict(intervals=intervals, seconds=len(windows)*5)
        for model in scores:
            rows = []
            for window in windows:
                item = window['historic'] if model=='historical_direct' else json.loads(
                    (OUT / 'annotations' / f'{window["name"]}.json').read_text())
                rows.append({'variants':{'selected':item}})
            probability = teacher_probability(rows,'selected',width)
            score = dict(name=name,group=inventory[name]['group'],seconds=len(windows)*5,
                area={'full':area(probability[:,kept],truth[:,kept])})
            scores[model][windows[0]['partition']].append(score)
    thresholds = {model:calibrate(parts['calibration']) for model,parts in scores.items()}
    summary = {model:summarize(parts['evaluation'],thresholds[model])['full'] for model,parts in scores.items()}
    fixed = {model:summarize(parts['calibration']+parts['evaluation'],{'full':1})['full'] for model,parts in scores.items()}
    attempts = [json.loads(p.read_text()) for p in (OUT / 'attempts').glob('*.json')]
    first,last = min(a['started_unix'] for a in attempts),max(a['started_unix']+a['elapsed_seconds'] for a in attempts)
    tokens = sum((a['usage'] or {}).get('completion_tokens',0) for a in attempts)
    timing = dict(attempts=len(attempts),batch_seconds=last-first,completion_tokens=tokens,
        aggregate_completion_tokens_per_second=tokens/(last-first),wall_seconds_per_window=(last-first)/100,
        mean_request_seconds=float(np.mean([a['elapsed_seconds'] for a in attempts])),
        generated_reasoning_characters=sum(a['reasoning_characters'] for a in attempts))
    result = dict(manifest_sha256=digest(OUT / 'manifest.json'), summary=summary, thresholds=thresholds,
        coverage=coverage, per_recording=scores, evaluation_windows=60, evaluation_seconds=300,
        calibration_windows=40, calibration_seconds=200, timing=timing,
        all_100_fixed_threshold_0p01_diagnostic=fixed,
        limitation='historical direct is an ambiguous-mode comparator, not a certified reasoning-enabled treatment')
    write_json(OUT / 'comparison.json',result)
    lines = ['# Preliminary matched no-reasoning pilot','',
        '| Teacher | Pixel AP | 2D IoU | Pixel precision | Pixel recall | Threshold |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,s in summary.items():
        lines.append(f'| {name} | {s["ap"]:.4f} | {s["iou"]:.4f} | {s["pooled_precision"]:.4f} | {s["pooled_recall"]:.4f} | {s["threshold"]:.2f} |')
    lines += ['', 'Report: 60 five-second windows from Recording_1. Thresholds selected separately on the same 40 windows from Recordings_2–4.',
        'Segment-macro scores over selected intervals only; not comparable to full-Powdermill tables. Historical direct mode was not verified.',
        '',f'100 windows completed in {timing["batch_seconds"]:.1f} s, including retries; {timing["aggregate_completion_tokens_per_second"]:.1f} aggregate output tokens/s. Server startup excluded.']
    (OUT / 'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,timing=timing),indent=2),flush=True)


def run(plan):
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPUs occupied; do not interfere')
    subprocess.run(['systemctl','--user','start',SERVER],check=True)
    try:
        deadline = time.monotonic()+180
        while True:
            try:
                ready = requests.get(URL+'/health',timeout=2).ok
            except requests.RequestException:
                ready = False
            if ready: break
            if time.monotonic()>deadline: raise ValueError('server startup timed out')
            time.sleep(2)
        props = requests.get(URL+'/props',timeout=10).json()
        if props['total_slots']!=32: raise ValueError('unexpected slot configuration')
        write_json(OUT / 'server_props.json',props)
        check = requests.post(URL+'/apply-template',json=payload(plan['windows'][0],plan,4096),timeout=30)
        check.raise_for_status()
        prompt = check.json()['prompt']
        if not re.search(r'<think>\s*</think>\s*$',prompt):
            raise ValueError('server did not render an empty closed thinking prefix')
        write_json(OUT / 'template_verification.json',dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            suffix=prompt[-100:],empty_closed_thinking_prefix=True))
        abort = threading.Event()
        write_json(OUT / 'status.json',dict(state='annotating',completed=0,total=100))
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = {pool.submit(annotate,w,plan,abort):w for w in plan['windows']}
            for index,future in enumerate(as_completed(futures),1):
                result = future.result()
                print(f'{index}/100 verified: {futures[future]["name"]}, {result["usage"]}',flush=True)
                write_json(OUT / 'status.json',dict(state='annotating',completed=index,total=100))
    finally:
        subprocess.run(['systemctl','--user','stop',SERVER],check=True)
    verify(plan)
    evaluate(plan)
    write_json(OUT / 'status.json',dict(state='complete',completed=100,total=100))
    print('Pilot complete. Original Qwen annotation queue remains paused.',flush=True)


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.prepare:
        print(f'Frozen {len(plan["windows"])} matched windows: 40 calibration, 60 evaluation.')
    else:
        try:
            run(plan)
        except Exception as error:
            write_json(OUT / 'status.json',dict(state='failed',error=str(error)))
            raise
