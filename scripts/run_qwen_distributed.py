#!/usr/bin/env python3
"""Coordinated backfill: unchanged Qwen requests, shared non-overwriting work claims."""
import argparse
import fcntl
import json
import os
import re
import signal
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from types import SimpleNamespace

import requests

import run_qwen_full_powdermill as full

ROLE=None
BACKEND=None
BACKEND_HASH=None
RUNTIME_HASH=None


def claim(condition,name):
    path=full.OUT/'backfill/locks'/condition/f'{name}.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a')
    try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close(); return None
    return handle


def annotate(task,plan,handle):
    window,condition,mode=task
    path=full.OUT/'annotations'/condition/f'{window["name"]}.json'
    existed=path.exists()
    try:
        full.annotate(window,condition,mode,plan)
        if not existed and path.exists():
            saved=json.loads(path.read_text())
            saved.update(worker_role=ROLE,worker_backend_sha256=BACKEND_HASH,worker_runtime_sha256=RUNTIME_HASH)
            full.study.write(path,saved)
        return path.exists()
    finally:
        # Every participant keeps the same inode; never unlink active claim files.
        fcntl.flock(handle,fcntl.LOCK_UN); handle.close()


def stage(tasks,plan,name):
    if full.study.digest(__file__)!=RUNTIME_HASH or full.study.digest(full.OUT/'backfill'/ROLE/'backend.json')!=BACKEND_HASH:
        raise ValueError('distributed runtime or backend configuration changed')
    folder=full.OUT/'backfill'/ROLE
    started=time.time(); completed=0; total=len(tasks); todo=deque(tasks); active={}
    log=folder/'batches'/f'{name}-{time.time_ns()}.json'
    record=dict(role=ROLE,stage=name,total=total,started_unix=started,completed=0,
        worker_backend_sha256=BACKEND_HASH,worker_runtime_sha256=RUNTIME_HASH)
    full.study.write(log,record)
    with ThreadPoolExecutor(max_workers=BACKEND['workers']) as pool:
        try:
            while todo or active:
                scan=len(todo)
                while todo and len(active)<BACKEND['workers'] and scan and not full.engine.STOP.is_set():
                    task=todo.popleft(); scan-=1
                    handle=claim(task[1],task[0]['name'])
                    if handle is None: todo.append(task); continue
                    active[pool.submit(annotate,task,plan,handle)]=task
                if not active:
                    if full.engine.STOP.is_set(): break
                    time.sleep(.5); continue
                done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    task=active.pop(future)
                    if future.result():
                        completed+=1
                        print(f'[{ROLE}] {name}: {completed}/{total} {task[0]["name"]}',flush=True)
                    elif not full.engine.STOP.is_set(): todo.append(task)
                record.update(completed=completed,active=len(active),pending=len(todo),updated_unix=time.time(),
                    state='pausing' if full.engine.STOP.is_set() else 'annotating')
                full.study.write(folder/'status.json',record)
                if ROLE=='twins': full.study.write(full.OUT/'status.json',record)
        except BaseException:
            full.engine.STOP.set(); raise
        finally:
            # The executor drains calls before the surrounding function returns.
            record.update(completed=completed,ended_unix=time.time()); full.study.write(log,record)
    if full.engine.STOP.is_set(): raise InterruptedError('paused after in-flight requests finished')
    if completed!=total: raise RuntimeError('incomplete stage; no missing windows are scored')


def helper_post(url,**kwargs):
    # CPU offload can increase latency; this changes only transport patience.
    if url.endswith('/v1/chat/completions'): kwargs['timeout']=BACKEND['request_timeout_seconds']
    return requests.post(url,**kwargs)


def helper_preflight(plan,window,condition,mode):
    thinking=not condition.startswith('direct_')
    settings=plan['reasoning' if thinking else 'no_reasoning']
    payload,_=full.engine.request(window,condition,mode,plan,settings['max_tokens'])
    response=requests.post(full.engine.URL+'/apply-template',json=payload,timeout=30)
    response.raise_for_status(); prompt=response.json()['prompt']
    expected=r'<think>\s*$' if thinking else r'<think>\s*</think>\s*$'
    if not re.search(expected,prompt) or thinking and 'Reasoning effort is set to xhigh.' not in prompt:
        raise ValueError('helper template does not match the frozen reasoning mode')
    full.study.write(full.OUT/'backfill'/ROLE/'template_checks'/f'{condition}-{mode}-{time.time_ns()}.json',
        dict(thinking=thinking,mode=mode,suffix=prompt[-100:],worker_backend_sha256=BACKEND_HASH))


def run_helper(plan):
    folder=full.OUT/'backfill'/ROLE
    response=requests.get(full.engine.URL+'/health',timeout=5); response.raise_for_status()
    props=requests.get(full.engine.URL+'/props',timeout=10).json()
    assert props['total_slots']==BACKEND['workers']
    assert props['default_generation_settings']['n_ctx']==16384
    assert props['model_path']==BACKEND['remote_model_path']
    full.study.write(folder/f'server_props-{time.time_ns()}.json',props)
    start,end=BACKEND['window_range']
    windows=plan['windows'][start:end]
    assert len(windows)==1155
    for condition in full.CONDITIONS:
        mode='plain' if condition=='direct_plain' else 'axes'
        with full.rendered(windows[0],mode) as (window,_):
            helper_preflight(plan,window,condition,mode)
        stage([(w,condition,mode) for w in windows],plan,condition)
        full.study.verify(plan)
    full.study.write(folder/'status.json',dict(state='complete',role=ROLE,windows=len(windows),
        conditions=full.CONDITIONS,worker_backend_sha256=BACKEND_HASH,worker_runtime_sha256=RUNTIME_HASH))


def main():
    global ROLE,BACKEND,BACKEND_HASH,RUNTIME_HASH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=['twins','viewing','vector'],required=True)
    args=parser.parse_args(); ROLE=args.role
    os.chdir(full.ROOT)
    for sig in [signal.SIGINT,signal.SIGTERM]: signal.signal(sig,lambda *_:full.engine.STOP.set())
    backend_file=full.OUT/'backfill'/ROLE/'backend.json'
    BACKEND=json.loads(backend_file.read_text()); BACKEND_HASH=full.study.digest(backend_file)
    RUNTIME_HASH=full.study.digest(__file__)
    assert BACKEND['full_manifest_sha256']==full.study.digest(full.OUT/'manifest.json')
    assert BACKEND['coordinator_sha256']==RUNTIME_HASH
    assert BACKEND['context_per_slot']==16384
    plan=full.prepare(); full.configure(); full.import_pilot(plan)
    full.engine.stage=stage
    if ROLE!='twins':
        full.engine.URL=BACKEND['url']
        full.engine.requests=SimpleNamespace(get=requests.get,post=helper_post)
    try:
        if ROLE=='twins': full.run(plan)
        else: run_helper(plan)
    except BaseException as error:
        status=dict(state='paused' if isinstance(error,InterruptedError) else 'failed',error=str(error),role=ROLE)
        full.study.write(full.OUT/'backfill'/ROLE/'status.json',status)
        if ROLE=='twins': full.study.write(full.OUT/'status.json',status)
        raise


if __name__=='__main__': main()
