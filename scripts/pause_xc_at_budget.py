#!/usr/bin/env python3
"""Pause both XC clients at a validated shared budget, draining in-flight calls."""
import argparse
import collections
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from qwen_prompt_study import write
from run_qwen_xc_50k import validate

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'data/annotations/xcl/powdermill_protocol_50000s_2026-09-14'
CLIENTS = [f'birdsong-qwen-xc50k-{role}-20260914.service' for role in ['twins', 'v100']]
SERVER = 'birdsong-qwen-xc50k-server-20260914.service'


def main(seconds, output):
    if seconds <= 0 or seconds % 5:
        raise ValueError('budget must be a positive multiple of five seconds')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = QUEUE / 'manifest.json'
    manifest_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = json.loads(source.read_text())
    if seconds > plan['target_seconds']:
        raise ValueError('budget exceeds the immutable annotation queue')
    windows = {w['name']: w for w in plan['windows']}
    accepted = {}

    def count():
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest_sha:
            raise ValueError('annotation manifest changed')
        for path in (QUEUE / 'annotations/self_review_1').glob('*.json'):
            if path.stem in accepted:
                continue
            window = windows[path.stem]
            if window['tile'][-1] - window['tile'][-2] != 1000:
                raise ValueError('expected a five-second window')
            if not all(validate(QUEUE, window, manifest_sha, c) for c in plan['conditions']):
                raise ValueError('incomplete annotation pair: ' + path.stem)
            accepted[path.stem] = json.loads(path.read_text())['worker_role']
        if len({windows[n]['tile'][0] for n in accepted}) != len(accepted):
            raise ValueError('duplicate source recordings')
        roles = collections.Counter(accepted.values())
        return dict(completed_windows=len(accepted), completed_seconds=5 * len(accepted),
                    by_worker_seconds={role: 5 * n for role, n in roles.items()})

    def status(state, **extra):
        value = dict(state=state, updated_unix=time.time(), target_seconds=seconds,
                     manifest_sha256=manifest_sha, **count(), **extra)
        write(output / 'status.json', value)
        return value

    while True:
        value = status('monitoring')
        if value['completed_seconds'] >= seconds:
            break
        time.sleep(20)

    status('draining', note='No new calls; accepted in-flight calls may exceed the target.')
    subprocess.run(['systemctl', '--user', 'stop', '--no-block', *CLIENTS], check=True)
    deadline = time.monotonic() + 7800
    while any(subprocess.check_output(['systemctl', '--user', 'show', unit,
                                      '-p', 'MainPID', '--value'], text=True).strip() != '0'
              for unit in CLIENTS):
        if time.monotonic() > deadline:
            raise TimeoutError('clients did not finish draining; inspect before stopping the server')
        status('draining')
        time.sleep(10)
    subprocess.run(['systemctl', '--user', 'stop', SERVER], check=True)
    value = status('paused_at_budget', note='V100 model server untouched; both annotation clients paused.')
    write(output / 'completed.json', dict(**value, windows=sorted(accepted),
          extra_seconds=value['completed_seconds'] - seconds))
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.seconds, args.output.resolve())
