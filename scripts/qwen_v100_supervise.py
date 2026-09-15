#!/usr/bin/env python3
"""Restart only the home Qwen server; record memory and guard against host OOM."""
import collections
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path('/media/george/DATA/birdsong-powdermill-worker-20260913')
STOP = threading.Event()


def status(record):
    record['updated_unix'] = time.time()
    temporary = ROOT / 'supervisor_status.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(ROOT / 'supervisor_status.json')


def memory(path):
    return {key: int(value.split()[0]) for line in Path(path).read_text().splitlines()
            if (key := line.split(':', 1)[0]) in {'RssAnon', 'RssFile', 'RssShmem', 'VmRSS', 'MemAvailable'}
            for value in [line.split(':', 1)[1]]}


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def main():
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, lambda *_: STOP.set())
    starts = collections.deque()
    while not STOP.is_set():
        now = time.time()
        while starts and starts[0] < now - 3600:
            starts.popleft()
        if len(starts) >= 3:
            status(dict(state='failed', error='Three server starts within one hour; manual inspection required.'))
            raise RuntimeError('Server recovery rate limit reached')
        starts.append(now)
        record = dict(state='running', started_unix=now, starts_last_hour=len(starts),
                      server_log=str(ROOT / f'logs/server-recovery-{int(now)}.log'))
        with open(record['server_log'], 'a', buffering=1) as log:
            child = subprocess.Popen(['bash', str(ROOT / 'qwen_v100_server.sh')],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            print(f'Started Qwen PID {child.pid}; log {record["server_log"]}', flush=True)
            next_sample = 0
            forced_restart = False
            try:
                while child.poll() is None and not STOP.is_set():
                    try:
                        record['memory_kib'] = memory(f'/proc/{child.pid}/status')
                    except FileNotFoundError:
                        break
                    record['host_memory_kib'] = memory('/proc/meminfo')
                    # File-backed model pages are reclaimable; guard anonymous allocations.
                    anon = record['memory_kib'].get('RssAnon', 0) + record['memory_kib'].get('RssShmem', 0)
                    if anon > 10 * 1024**2:
                        record.update(state='ram_guard', error='Anonymous server memory exceeded 10 GiB.')
                        forced_restart = True
                        status(record)
                        print(record['error'], flush=True)
                        break
                    if time.monotonic() >= next_sample:
                        try:
                            sample = subprocess.run(['nvidia-smi', '--query-gpu=index,temperature.gpu,memory.used,utilization.gpu,power.draw',
                                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                            record['gpu_csv'] = sample.stdout.strip()
                            record['gpu_query_error'] = sample.stderr.strip()
                        except subprocess.TimeoutExpired:
                            record['gpu_query_error'] = 'nvidia-smi timed out'
                        with (ROOT / 'logs/ram-recovery-telemetry.jsonl').open('a') as telemetry:
                            telemetry.write(json.dumps({**record, 'sampled_unix': time.time()}) + '\n')
                        next_sample = time.monotonic() + 30
                    status(record)
                    STOP.wait(10)
            except Exception as error:
                forced_restart = True
                record['error'] = repr(error)
            finally:
                stop_child(child)
            record.update(state='stopped' if STOP.is_set() or child.returncode == 0 and not forced_restart else 'restarting',
                          exit_code=child.returncode)
            status(record)
            print(f'Qwen exited {child.returncode}; {record["state"]}', flush=True)
            if record['state'] == 'stopped':
                return
        STOP.wait(60)


if __name__ == '__main__':
    main()
