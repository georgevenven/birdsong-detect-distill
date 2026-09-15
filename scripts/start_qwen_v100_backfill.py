#!/usr/bin/env python3
"""Gate the V100 helper on readiness and isolated real annotation requests."""
import hashlib
import json
import os
import re
import sys
import time

import requests

import run_qwen_full_powdermill as full


def main():
    os.chdir(full.ROOT)
    folder = full.OUT / 'backfill/vector'
    backend_path = folder / 'backend.json'
    backend = json.loads(backend_path.read_text())
    backend_hash = full.study.digest(backend_path)
    runtime_hash = full.study.digest(__file__)
    url = backend['url']
    deadline = time.monotonic() + 43200
    print('Waiting for the detached V100 build, verified download and server.', flush=True)
    while time.monotonic() < deadline:
        try:
            response = requests.get(url + '/health', timeout=5)
            if response.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(15)
    else:
        raise TimeoutError('V100 readiness deadline exceeded')
    response = requests.get(url + '/props', timeout=30)
    response.raise_for_status()
    props = response.json()
    assert props['model_path'] == backend['remote_model_path']
    assert props['total_slots'] == backend['workers']
    assert props['default_generation_settings']['n_ctx'] == 16384
    full.study.write(folder / f'diagnostics/server_props-{time.time_ns()}.json', props)
    assert full.study.digest(full.ROOT / 'scripts/run_qwen_distributed.py') == backend['coordinator_sha256']
    ready_path = folder / 'diagnostics/ready.json'
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        if ready.get('backend_sha256') == backend_hash and ready.get('preflight_runtime_sha256') == runtime_hash:
            print('Reusing successful preflight for this exact backend/runtime.', flush=True)
            os.execv(sys.executable, [sys.executable, '-u', 'scripts/run_qwen_distributed.py', '--role', 'vector'])
    plan = full.prepare()
    full.configure()
    full.import_pilot(plan)
    # Reused pilot calibration window: no canonical annotation is overwritten.
    window = next(w for w in plan['windows'] if w['name'] == 'Recording_2_Segment_03_38000_39000')
    for condition in ['direct_axes', 'reasoning', 'self_review_2']:
        thinking = condition != 'direct_axes'
        settings = plan['reasoning' if thinking else 'no_reasoning']
        with full.rendered(window, 'axes') as (ready, picture):
            payload, parent_hash = full.engine.request(ready, condition, 'axes', plan, settings['max_tokens'])
            request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            applied = requests.post(url + '/apply-template', json=payload, timeout=30)
            applied.raise_for_status()
            prompt = applied.json()['prompt']
            ending = r'<think>\s*$' if thinking else r'<think>\s*</think>\s*$'
            assert re.search(ending, prompt)
            assert not thinking or 'Reasoning effort is set to xhigh.' in prompt
            print(f'V100 preflight: {condition}', flush=True)
            started = time.monotonic()
            response = requests.post(url + '/v1/chat/completions', json=payload, timeout=7200)
            response.raise_for_status()
            raw = response.json()
            choice = raw['choices'][0]
            message = choice['message']
            reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
            content = message.get('content') or ''
            reason_tokens = raw.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')
            mode_ok = bool(reasoning) if thinking else not reasoning and reason_tokens in (None, 0)
            mode_ok = mode_ok and '<think>' not in content and '</think>' not in content
            record = dict(condition=condition, window=window['name'], request_sha256=request_hash,
                input_image_sha256=full.study.digest(picture), parent_annotation_sha256=parent_hash,
                elapsed_seconds=time.monotonic() - started, usage=raw.get('usage'), timings=raw.get('timings'),
                finish_reason=choice['finish_reason'], reasoning_characters=len(reasoning),
                content=content, mode_verified=mode_ok, worker_backend_sha256=backend_hash,
                note='Deployment preflight only; excluded from canonical annotations and scores.')
            full.study.write(folder / f'diagnostics/{condition}-{time.time_ns()}.json', record)
            assert mode_ok and choice['finish_reason'] == 'stop'
            full.study.parsed_events(json.loads(content), window)
    assert full.study.digest(backend_path) == backend_hash
    full.study.verify(plan)
    full.study.write(ready_path, dict(backend_sha256=backend_hash, preflight_runtime_sha256=runtime_hash, verified_unix=time.time()))
    print('V100 preflight passed; starting locked fourth-quarter backfill.', flush=True)
    os.execv(sys.executable, [sys.executable, '-u', 'scripts/run_qwen_distributed.py', '--role', 'vector'])


if __name__ == '__main__':
    main()
