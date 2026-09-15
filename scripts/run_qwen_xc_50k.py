#!/usr/bin/env python3
"""Detached two-backend XC annotation with the unchanged Powdermill request builder."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import requests

import qwen_prompt_study as study
import run_qwen_prompt_study as engine

LOCAL = threading.local()


def validate(out, window, plan_hash, condition):
    path = out/'annotations'/condition/f'{window["name"]}.json'
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    assert saved['accepted'] and saved['mode_verified'] and saved['mode'] == 'axes'
    assert saved['manifest_sha256'] == plan_hash and saved['input_spectrogram_sha256'] == window['spectrogram_sha256']
    assert saved['condition'] == condition and saved['window'] == window['name']
    if condition == 'self_review_1':
        assert saved['parent_annotation_sha256'] == study.digest(out/'annotations/reasoning'/path.name)
    else:
        assert saved['parent_annotation_sha256'] is None
    study.parsed_events(saved['raw_annotation'], window)
    return True


def export(out, plan, plan_hash):
    handle = (out/'export.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return
    try:
        if (out/'complete.json').exists():
            return
        for condition in plan['conditions']:
            destination = out/f'{condition}.jsonl'
            temporary = destination.with_suffix('.jsonl.tmp')
            with temporary.open('w') as stream:
                stream.write(json.dumps(dict(type='metadata', variant=condition, seconds=plan['target_seconds'],
                    manifest_sha256=plan_hash, protocol='Powdermill reasoning + one numbered self-review'))+'\n')
                for window in plan['windows']:
                    assert validate(out, window, plan_hash, condition)
                    path = out/'annotations'/condition/f'{window["name"]}.json'
                    annotation = json.loads(path.read_text())
                    name, shard, start, end, left, right = window['tile']
                    row = dict(status='ok', recording=name, source=dict(shard=shard,start=start,end=end),
                        tile=dict(start_timebin=left,end_timebin=right,ownership_start_timebin=left,ownership_end_timebin=right),
                        events=annotation['events'], annotation=str(path), annotation_sha256=study.digest(path),
                        manifest_sha256=plan_hash, worker_role=annotation['worker_role'],
                        worker_backend_sha256=annotation['worker_backend_sha256'])
                    stream.write(json.dumps(row,separators=(',',':'))+'\n')
            temporary.replace(destination)
        study.write(out/'complete.json', dict(state='complete', completed_unix=time.time(),
            windows=len(plan['windows']), seconds=plan['target_seconds'], manifest_sha256=plan_hash,
            exports={c:study.digest(out/f'{c}.jsonl') for c in plan['conditions']}))
    finally:
        handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--role', choices=['twins','v100'], required=True)
    args = parser.parse_args()
    os.chdir(study.ROOT)
    out = args.out.resolve(); plan = json.loads((out/'manifest.json').read_text())
    plan_hash = study.digest(out/'manifest.json'); backend = plan['backends'][args.role]
    backend_hash = hashlib.sha256(json.dumps(backend,sort_keys=True).encode()).hexdigest()
    folder = out/'workers'/args.role; folder.mkdir(parents=True,exist_ok=True)
    (out/'locks').mkdir(exist_ok=True)
    # Prevent an accidental second client for one API, without blocking the other backend.
    own_lock = (folder/'client.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGINT,signal.SIGTERM]:
        signal.signal(sig,lambda *_:engine.STOP.set())
    for name, sha in plan['protected'].items():
        if study.digest(name) != sha:
            raise ValueError('frozen input changed: ' + name)
    engine.OUT = out; engine.URL = backend['url']
    original_write = engine.write
    def annotated_write(path, value):
        extra = getattr(LOCAL,'provenance',{})
        original_write(path, {**value,**extra})
    engine.write = annotated_write
    def post(url, **kwargs):
        kwargs['timeout'] = backend['request_timeout_seconds']
        return requests.post(url, **kwargs)
    engine.requests = SimpleNamespace(post=post)
    checked, preflight_lock = set(), threading.Lock()
    def preflight(window, condition):
        with preflight_lock:
            if condition in checked:
                return
            payload, _ = engine.request(window,condition,'axes',plan,plan['reasoning']['max_tokens'])
            response = requests.post(backend['url']+'/apply-template',json=payload,timeout=30)
            response.raise_for_status(); prompt = response.json()['prompt']
            if not re.search(r'<think>\s*$',prompt) or 'Reasoning effort is set to xhigh.' not in prompt:
                engine.STOP.set()
                raise ValueError('server did not apply the frozen xhigh reasoning template')
            study.write(folder/'template_checks'/f'{condition}.json',dict(suffix=prompt[-100:],
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),backend_sha256=backend_hash))
            checked.add(condition)
    def process(window, handle):
        try:
            if validate(out,window,plan_hash,'self_review_1'):
                return False
            scratch = out/'scratch'; scratch.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=args.role+'-',dir=scratch) as temporary:
                spec, _, _ = study.context(Path(plan['spec_dir']),window['tile'],*window['tile'][-2:])
                assert hashlib.sha256(spec.tobytes(order='C')).hexdigest() == window['spectrogram_sha256']
                picture = Path(temporary)/'input.png'
                study.decorate(study.image(spec),True).save(picture)
                ready = {**window,'images':{'axes':str(picture)}}
                LOCAL.provenance = dict(worker_role=args.role,worker_backend_sha256=backend_hash,
                    input_image_sha256=study.digest(picture),input_spectrogram_sha256=window['spectrogram_sha256'])
                for condition in plan['conditions']:
                    if engine.STOP.is_set():
                        return False
                    if not validate(out,window,plan_hash,condition):
                        preflight(ready,condition)
                        engine.annotate(ready,condition,'axes',plan)
                    if not engine.STOP.is_set():
                        assert validate(out,window,plan_hash,condition)
            return validate(out,window,plan_hash,'self_review_1')
        finally:
            handle.close()
    status = dict(role=args.role,state='waiting_for_api',started_unix=time.time(),manifest_sha256=plan_hash,
        backend_sha256=backend_hash,target_windows=len(plan['windows']),target_seconds=plan['target_seconds'])
    def update(**values):
        status.update(values,updated_unix=time.time()); study.write(folder/'status.json',status)
    update()
    deadline = time.monotonic()+43200
    while not engine.STOP.is_set():
        try:
            response = requests.get(backend['url']+'/health',timeout=5)
            if response.ok:
                break
        except requests.RequestException:
            pass
        if time.monotonic()>deadline:
            raise TimeoutError('backend not ready after twelve hours')
        engine.STOP.wait(15)
    if engine.STOP.is_set():
        update(state='paused'); return
    props = requests.get(backend['url']+'/props',timeout=15).json()
    assert props['model_path'] == backend['remote_model_path']
    assert props['total_slots'] == backend['workers'] == 16
    assert props['default_generation_settings']['n_ctx'] == backend['context_per_slot'] == 16384
    study.write(folder/f'server_props-{time.time_ns()}.json', props)
    own_completed = 0
    try:
        for attempt in range(3):
            todo = deque(plan['windows']); active = {}; failed = []
            with ThreadPoolExecutor(max_workers=backend['workers']) as pool:
                try:
                    while (todo or active) and not engine.STOP.is_set():
                        scan = len(todo)
                        while todo and len(active)<backend['workers'] and scan and not engine.STOP.is_set():
                            window = todo.popleft(); scan -= 1
                            handle = (out/'locks'/f'{window["name"]}.lock').open('a')
                            try:
                                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except BlockingIOError:
                                handle.close(); todo.append(window); continue
                            active[pool.submit(process,window,handle)] = window
                        if not active:
                            engine.STOP.wait(2); continue
                        done, _ = wait(active,timeout=1,return_when=FIRST_COMPLETED)
                        for future in done:
                            window = active.pop(future)
                            try:
                                if future.result():
                                    own_completed += 1
                                    print(f'{args.role}: completed {window["name"]}; new pairs this process {own_completed}',flush=True)
                            except (requests.RequestException, AssertionError):
                                engine.STOP.set(); raise
                            except Exception as error:
                                failed.append(window['name'])
                                study.write(folder/'failures'/f'{window["name"]}-{time.time_ns()}.json',
                                    dict(window=window['name'],error=repr(error),pass_index=attempt))
                                if engine.STOP.is_set() or len(failed)>=10:
                                    engine.STOP.set(); raise
                        update(state='annotating',active=len(active),pending=len(todo),new_pairs=own_completed,
                            failed_this_pass=len(failed),pass_index=attempt)
                except BaseException:
                    engine.STOP.set(); raise  # Executor drains accepted in-flight calls before exit.
            if engine.STOP.is_set():
                update(state='paused',active=0); return
            if not failed:
                export(out,plan,plan_hash)
                update(state='complete',active=0,pending=0,completed_seconds=plan['target_seconds'])
                return
        raise RuntimeError('some windows remain invalid after three bounded passes; inspect saved attempts')
    except BaseException as error:
        update(state='failed',error=repr(error),active=0)
        raise


if __name__ == '__main__':
    main()
