#!/usr/bin/env python3
"""One detached readiness/provenance check after the viewing helper's graceful restart."""
import json
import subprocess
import time

import requests

import run_qwen_full_powdermill as full


def service(role):
    unit = ('birdsong-qwen-full-powdermill-20260911.service' if role == 'twins'
            else 'birdsong-qwen-viewing-backfill-20260911.service')
    output = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'MainPID'], text=True)
    return dict(line.split('=', 1) for line in output.splitlines())


def main():
    folder = full.OUT / 'backfill'
    request = json.loads((folder / 'overnight_request.json').read_text())
    report = folder / 'overnight_verification.json'
    full.study.write(report, dict(state='waiting_for_viewing_restart', started_unix=time.time()))
    while time.time() < request['deadline_unix']:
        statuses = {role: service(role) for role in ['twins', 'viewing']}
        ready = all(s['ActiveState'] == 'active' for s in statuses.values())
        ready &= statuses['viewing']['MainPID'] != str(request['previous_viewing_pid'])
        details = {}
        if ready:
            try:
                for role in statuses:
                    backend = json.loads((folder / role / 'backend.json').read_text())
                    response = requests.get(backend['url'] + '/props', timeout=10)
                    response.raise_for_status()
                    props = response.json()
                    assert props['model_path'] == backend['remote_model_path']
                    assert props['total_slots'] == backend['workers']
                    assert props['default_generation_settings']['n_ctx'] == 16384
                    if role == 'viewing': assert backend['request_timeout_seconds'] == 7200
                    status = json.loads((folder / role / 'status.json').read_text())
                    assert status['worker_backend_sha256'] == full.study.digest(folder / role / 'backend.json')
                    slots = requests.get(backend['url'] + '/slots', timeout=10)
                    slots.raise_for_status()
                    active = sum(slot['is_processing'] for slot in slots.json())
                    ready &= active > 0
                    details[role] = dict(**statuses[role], active_requests=active,
                        context_per_slot=16384, workers=backend['workers'],
                        request_timeout_seconds=backend['request_timeout_seconds'])
            except (requests.RequestException, ValueError, KeyError, AssertionError):
                ready = False
        if ready:
            full.prepare()  # Recheck the frozen input and protocol hashes.
            previous = json.loads((folder / 'pre_handoff_sha256.json').read_text())
            for name, expected in previous['sha256'].items():
                assert full.study.digest(full.OUT / name) == expected, name
            full.study.write(report, dict(state='verified', checked_unix=time.time(),
                services=details, preserved_annotations=len(previous['sha256']),
                note='Detached readiness check only; not a claim that annotation is complete.'))
            print('Twins and restarted viewing helper verified active; prior labels unchanged.', flush=True)
            return
        time.sleep(15)
    raise TimeoutError('Workers did not pass the post-restart check before the deadline')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        full.study.write(full.OUT / 'backfill/overnight_verification.json',
            dict(state='failed', error=str(error), checked_unix=time.time()))
        raise
