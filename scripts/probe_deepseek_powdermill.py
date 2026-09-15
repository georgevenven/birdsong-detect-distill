#!/usr/bin/env python3
"""Small, audited Zen capability probe; never starts the annotation batch."""
import argparse
import getpass
import hashlib
import json
import os
import time
from pathlib import Path

import requests

import qwen_prompt_study as study
import run_qwen_prompt_study as qwen

OUT = study.ROOT / 'results/deepseek_teacher_powdermill/prompt_250_2026-09-13'
KEY = Path(f'/run/user/{os.getuid()}/birdsong-deepseek-pilot-20260913.key')
URL = 'https://opencode.ai/zen/v1/chat/completions'


def credential():
    if KEY.exists():
        if KEY.stat().st_mode & 0o077:
            raise ValueError('Credential permissions must be private')
        return KEY.read_text().strip()
    secret = getpass.getpass('OpenCode key (hidden): ').strip()
    if not secret or '\n' in secret:
        raise ValueError('Invalid credential')
    fd = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(secret)
    return secret


def payload(window, condition, mode, plan, model):
    thinking = not condition.startswith('direct_')
    limit = 8192 if thinking else 4096
    request, parent = qwen.request(window, condition, mode, plan, limit)
    for name in ['top_k', 'seed', 'reasoning_budget_tokens', 'reasoning_format', 'chat_template_kwargs']:
        request.pop(name, None)
    request.update(model=model, thinking={'type': 'enabled' if thinking else 'disabled'},
                   reasoning_effort='high' if thinking else 'none', response_format={'type': 'json_object'})
    if thinking:
        request.pop('temperature', None)  # DeepSeek ignores it in thinking mode.
    else:
        request.pop('top_p', None)  # DeepSeek fixes non-thinking top_p to 1.
    return request, parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='deepseek-v4-flash-vision-exp')
    parser.add_argument('--condition', choices=['direct_axes', 'reasoning'], default='direct_axes')
    parser.add_argument('--max-tokens', type=int, choices=[4096, 8192, 16384, 32768])
    args = parser.parse_args()
    plan = study.prepare()
    window = next(w for w in plan['windows'] if w['partition'] == 'calibration')
    request, _ = payload(window, args.condition, 'axes', plan, args.model)
    if args.max_tokens:
        request['max_tokens'] = args.max_tokens
    secret = credential()
    folder = OUT / 'capability_probes'
    if sum(1 for _ in folder.glob('*.json')) >= 4:
        raise ValueError('Four-probe safety limit reached; inspect existing results')
    started = time.time()
    audit = dict(started_unix=started, window=window['name'], condition=args.condition,
                 requested_model=args.model, endpoint=URL, max_tokens=request['max_tokens'],
                 request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                 image_sha256=study.digest(window['images']['axes']),
                 source_manifest_sha256=study.digest(study.OUT / 'manifest.json'))
    try:
        response = requests.post(URL, json=request, timeout=(20, 600), allow_redirects=False,
                                 headers={'Authorization': 'Bearer ' + secret,
                                          'User-Agent': 'birdsong-research-pilot/1.0'})
        audit.update(http_status=response.status_code, elapsed_seconds=time.time() - started,
                     provider_metadata={name: response.headers[name] for name in
                         ['x-opencode-endpoint-id', 'x-opencode-upstream-model-id',
                          'x-request-id', 'x-ds-request-id'] if name in response.headers})
        try:
            raw = response.json()
        except ValueError:
            raw = {'error': response.text[:2000]}
        audit['response'] = raw
        # Keep error messages useful without ever recording an echoed credential.
        if secret in json.dumps(raw):
            audit['response'] = {'error': 'Response unexpectedly echoed credential; withheld'}
            raise ValueError('Credential echoed by server')
        if response.ok:
            choice = raw['choices'][0]
            message = choice['message']
            reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
            audit.update(returned_model=raw.get('model'), finish_reason=choice.get('finish_reason'),
                         reasoning_characters=len(reasoning), usage=raw.get('usage'))
            try:
                result = json.loads(message.get('content') or '')
                audit['events'] = len(study.parsed_events(result, window))
                audit['schema_valid'] = True
            except (ValueError, TypeError):
                audit['schema_valid'] = False
    except requests.RequestException as error:
        audit.update(error=type(error).__name__, elapsed_seconds=time.time() - started)
    finally:
        study.write(folder / f'{time.time_ns()}.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in ['response']}, indent=2))
    if audit.get('http_status') != 200:
        print('API error:', json.dumps(audit.get('response', {}))[:1500])


if __name__ == '__main__':
    main()
