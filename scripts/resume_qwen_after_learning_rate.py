#!/usr/bin/env python3
"""Honor the requested delayed XC resumption without altering running experiment services."""
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from birdsong_detect_distill.benchmark_data import digest, write_json

OUT = Path('results/qwen_resume_after_lr_2026-09-10')
LR = Path('results/qwen_teacher_powdermill/learning_rate_10000train_800val_2026-09-10')
MARKER = Path('/home/george-vengrovski/.config/birdsong-detect-distill/qwen.paused')
SERVER = 'songmae-qwen.service'
CLIENT = 'birdsong-qwen-xc-continuation.service'
DEPENDENCIES = ['birdsong-yolo11l-external-20260910.service', 'birdsong-learning-rate-20260910.service']


def state(unit):
    output = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ExecMainStatus'], text=True)
    return dict(line.split('=', 1) for line in output.splitlines())


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    if (OUT / 'resumed.json').exists():
        raise ValueError('this one-time resumption has already completed')
    if not MARKER.exists() or any(state(unit)['MainPID'] != '0' for unit in [SERVER, CLIENT]):
        raise ValueError('expected Qwen to remain paused before scheduling')
    queue_manifest = Path('data/annotations/xcl/continuation_2026-09-09/manifest.json')
    continuation = json.loads(queue_manifest.read_text())
    master = Path('data/annotations/xcl/qwen38_adaptive_review_5s_annotations.jsonl')
    approval = dict(trigger=DEPENDENCIES, result=str(LR / 'summary.json'),
        marker_sha256=digest(MARKER), marker_mtime_ns=MARKER.stat().st_mtime_ns,
        queue=continuation['queue'], queue_manifest_sha256=digest(queue_manifest),
        master_prefix_bytes=master.stat().st_size, master_prefix_sha256=digest(master),
        script_sha256=digest(__file__), requested='Resume original XC annotations after the 3e-4 comparison completes')
    queued = OUT / 'queued.json'
    if queued.exists() and json.loads(queued.read_text()) != approval:
        raise ValueError('queued resumption context changed; inspect before retrying')
    if not queued.exists():
        write_json(queued, approval)
    print('Queued: wait for successful YOLO11l and LR training/evaluation/reporting; Qwen stays paused.', flush=True)
    while any(state(unit)['ActiveState'] in {'active', 'activating', 'deactivating', 'reloading'} for unit in DEPENDENCIES):
        time.sleep(30)
    if any(state(unit)['ExecMainStatus'] != '0' for unit in DEPENDENCIES):
        raise ValueError('an experiment failed; leaving Qwen paused')
    summary = json.loads((LR / 'summary.json').read_text())
    for model in summary['models'].values():
        if digest(model['checkpoint']) != model['checkpoint_sha256']:
            raise ValueError('LR checkpoint changed')
    for path, sha in summary['sources_sha256'].items():
        if digest(path) != sha:
            raise ValueError('LR report changed')
    if set(summary['models']) != {'1e-3', '3e-4'} or not (LR / 'comparison.csv').exists():
        raise ValueError('LR comparison is incomplete')
    if not Path('results/yolo11l_external_2026-09-10/tables/table_sources.json').exists():
        raise ValueError('YOLO tables are incomplete')
    print('Experiments complete; waiting for both GPUs to be free.', flush=True)
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
        time.sleep(30)
    if (not MARKER.exists() or digest(MARKER) != approval['marker_sha256']
            or MARKER.stat().st_mtime_ns != approval['marker_mtime_ns']):
        raise ValueError('pause state changed after scheduling; do not override a newer pause')
    if digest(continuation['queue']['path']) != continuation['queue']['sha256']:
        raise ValueError('frozen XC queue changed')
    if master.stat().st_size != approval['master_prefix_bytes'] or digest(master) != approval['master_prefix_sha256']:
        raise ValueError('XC annotations changed while paused')
    for path, sha in continuation['protected_files'].items():
        if digest(path) != sha:
            raise ValueError(f'protected experiment data changed: {path}')
    environment = dict(item.split('=', 1) for item in shlex.split(subprocess.check_output(
        ['systemctl', '--user', 'show', SERVER, '-p', 'Environment', '--value'], text=True)))
    if environment.get('PARALLEL') != '16' or environment.get('CTX_SIZE') != '262144':
        raise ValueError('expected 16 slots with 16,384 context tokens each')
    if shutil.disk_usage('.').free < 2 * 1024**3:
        raise ValueError('less than 2 GiB free; leaving Qwen paused')
    archived_marker = OUT / 'pause-marker-before-resume.txt'
    if archived_marker.exists():
        raise ValueError('resumption was already attempted; inspect before retrying')
    MARKER.rename(archived_marker)
    subprocess.run(['systemctl', '--user', 'start', SERVER], check=True)
    deadline = time.monotonic() + 300
    while True:
        try:
            with urlopen('http://127.0.0.1:8080/health', timeout=5) as response:
                health = json.load(response)
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError('Qwen server did not become healthy; annotation client was not started')
            time.sleep(2)
    with urlopen('http://127.0.0.1:8080/slots', timeout=10) as response:
        slots = json.load(response)
    if len(slots) != 16 or any(slot.get('n_ctx', 16384) != 16384 for slot in slots):
        raise ValueError('running Qwen slot configuration differs from requested context')
    subprocess.run(['systemctl', '--user', 'start', CLIENT], check=True)
    if state(CLIENT)['ActiveState'] != 'active':
        raise ValueError('XC annotation client did not start')
    write_json(OUT / 'resumed.json', dict(queued_sha256=digest(queued), health=health,
        slots=len(slots), context_per_slot=16384, server=state(SERVER), client=state(CLIENT),
        lr_summary_sha256=digest(LR / 'summary.json'), archived_pause_marker=str(archived_marker)))
    print('Resumed original XC queue: 16 workers, 16,384 tokens per server slot; existing checkpoints retained.', flush=True)


if __name__ == '__main__':
    main()
