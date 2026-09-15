#!/usr/bin/env python3
"""Budgeted Zen teacher pilot; reuses frozen Qwen inputs and scoring, never its jobs."""
import argparse
import fcntl
import hashlib
import json
import signal
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests

import probe_deepseek_powdermill as probe
import qwen_prompt_study as study
import run_qwen_prompt_study as qwen

OUT, SOURCE = probe.OUT, study.OUT
MODEL = 'deepseek-v4-flash-vision-exp'
CONDITIONS = ['direct_plain', 'direct_axes', 'reasoning', 'self_review_1', 'self_review_2']
STOP = threading.Event()
CAP = 5.50


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rates(stamp):
    date = datetime.fromtimestamp(stamp, timezone.utc)
    peak = date.weekday() < 5 and (1 <= date.hour < 4 or 6 <= date.hour < 10)
    return (.30, 1.20) if peak else (.15, .60)


def charge(usage, started, ended):
    # Higher than Zen's listed .14/.28 rates; ignore cache discounts conservatively.
    stamps = [started, ended] + list(range(int(started), int(ended) + 1, 900))
    inp, output = max(rates(t) for t in stamps)
    return (usage['prompt_tokens'] * inp + usage['completion_tokens'] * output) / 1e6


class Budget:
    """Durable reservations survive interruption; unknown charges never disappear."""
    def __init__(self):
        self.path = OUT / 'budget.sqlite3'
        self.condition = threading.Condition()
        self.active = 0
        self.not_before = 0
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, reserve REAL, charge REAL, state TEXT)')
            for path in sorted((OUT / 'capability_probes').glob('*.json')):
                item = read(path)
                usage = item.get('usage')
                reserved = (65536 * .30 + item['max_tokens'] * 1.20) / 1e6
                cost = charge(usage, item['started_unix'], item['started_unix'] + item['elapsed_seconds']) if usage else reserved
                if item.get('http_status') == 401:
                    cost = 0  # Explicit unsupported-model rejection, no generation.
                db.execute('INSERT OR IGNORE INTO calls VALUES (?, ?, ?, ?)',
                           ('probe:' + path.name, reserved, cost, 'probe'))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def totals(self):
        with self.connect() as db:
            row = db.execute('SELECT COALESCE(SUM(COALESCE(charge,reserve)),0), '
                             'COALESCE(SUM(charge),0), COUNT(*) FROM calls').fetchone()
        return dict(cap_usd=CAP, committed_upper_usd=row[0], accounted_upper_usd=row[1], calls=row[2])

    def reserve(self, name, maximum):
        with self.condition:
            while True:
                if STOP.is_set():
                    raise InterruptedError('Draining without new requests')
                if time.time() < self.not_before:
                    self.condition.wait(min(1, self.not_before - time.time()))
                    continue
                with self.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    used = db.execute('SELECT COALESCE(SUM(COALESCE(charge,reserve)),0) FROM calls').fetchone()[0]
                    if used + maximum <= CAP:
                        db.execute('INSERT INTO calls VALUES (?, ?, NULL, ?)', (name, maximum, 'reserved'))
                        self.active += 1
                        return
                if not self.active:
                    raise RuntimeError('Budget ceiling reached; completed labels retained')
                self.condition.wait(1)

    def settle(self, name, cost, state):
        with self.condition:
            with self.connect() as db:
                reserved = db.execute('SELECT reserve FROM calls WHERE id=?', (name,)).fetchone()[0]
                db.execute('UPDATE calls SET charge=?,state=? WHERE id=?', (cost, state, name))
            self.active -= 1
            self.condition.notify_all()
            if cost is not None and cost > reserved + 1e-9:
                STOP.set()
                raise RuntimeError('Provider usage exceeded reserved ceiling; manual billing review required')


def prepare():
    original = study.prepare()  # Validate Qwen's complete frozen input set before redirecting output.
    plan = dict(original)
    protected = {str(SOURCE / name): study.digest(SOURCE / name)
                 for name in ['manifest.json', 'review_amendment.json', 'axis_decision.json']}
    protected.update({str(path): study.digest(path) for path in
                      [Path(__file__), Path(probe.__file__), Path(study.__file__), Path(qwen.__file__)]})
    plan.update(model=MODEL, model_display='DeepSeek V4 Flash Vision Exp (OpenCode Zen)', endpoint=probe.URL,
                protected=protected, reference_qwen_settings={name: original[name] for name in
                    ['reasoning', 'no_reasoning', 'temperature', 'top_p', 'top_k', 'context_per_slot']},
                axis_selection='Axes frozen from the prior Qwen pilot; no new format selection',
                source_manifest_sha256=study.digest(SOURCE / 'manifest.json'),
                context_per_slot=None, workers=32, requested_max_tokens={'direct': [4096, 6144], 'thinking': [32768, 49152]},
                budget_usd=CAP, sampling_controls='DeepSeek high/none; direct temperature .5; thinking top_p .95; no seed or top_k',
                capability_differences='No verified V4.1 identity, no separate reasoning-token budget, JSON object not constrained schema',
                execution='Four calibration windows in all conditions, then 16-call and 32-call batches; bounded retries; no local GPU use',
                billing='Reserve peak .30/1.20 per million with conservative input bound; account .15/.60 off-peak or .30/1.20 peak, ignoring cache',
                caveat='Exploratory 250-window teacher control, not full Powdermill or student results. Model family settings are not identical.')
    path = OUT / 'manifest.json'
    if path.exists():
        frozen = read(path)
        amendment_path = OUT / 'runtime_amendment.json'
        expected = dict(frozen)
        if amendment_path.exists():
            amendment = read(amendment_path)
            if amendment['manifest_sha256'] != study.digest(path):
                raise ValueError('Runtime amendment has the wrong source manifest')
            expected['protected'] = dict(frozen['protected'])
            for file, change in amendment['replacements'].items():
                if frozen['protected'][file] != change['before'] or study.digest(change['snapshot']) != change['before']:
                    raise ValueError('Runtime amendment history mismatch')
                expected['protected'][file] = change['after']
        if expected != plan:
            raise ValueError('Frozen DeepSeek protocol changed; inspect before resuming')
    else:
        study.write(path, plan)
    study.OUT = qwen.OUT = OUT  # Process-local only; never edit the running Qwen code or outputs.
    return plan


def request(window, condition, plan, attempt=0):
    mode = condition.removeprefix('direct_') if condition.startswith('direct_') else 'axes'
    body, parent = probe.payload(window, condition, mode, plan, MODEL)
    key = 'direct' if condition.startswith('direct_') else 'thinking'
    body['max_tokens'] = plan['requested_max_tokens'][key][int(attempt > 0)]
    return body, parent, mode


def parse(raw, window, thinking):
    choice = raw['choices'][0]
    message = choice['message']
    reason = message.get('reasoning_content') or message.get('reasoning') or ''
    content = message.get('content') or ''
    usage = raw.get('usage') or {}
    reason_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    verified = bool(reason) or bool(reason_tokens) if thinking else not reason and reason_tokens in (None, 0)
    if not verified or '<think>' in content or '</think>' in content:
        raise RuntimeError('Reasoning mode mismatch; refusing to mix conditions')
    if raw.get('model') != MODEL:
        raise RuntimeError('Returned model changed; refusing to mix teachers')
    if choice['finish_reason'] != 'stop':
        raise ValueError('No complete final JSON: ' + str(choice['finish_reason']))
    result = json.loads(content)
    if isinstance(result, dict) and set(result) == {'type', 'events'} and result['type'] == 'object':
        result = {'events': result['events']}  # Harmless wrapper metadata; no detection edits.
    return result, study.parsed_events(result, window), len(reason)


def canonical(audit, raw, window, plan):
    result, events, reason_chars = parse(raw, window, not audit['condition'].startswith('direct_'))
    return dict(audit, accepted=True, manifest_sha256=study.digest(OUT / 'manifest.json'),
                raw_annotation=result, events=events, mode_verified=True, reasoning_characters=reason_chars,
                runtime_sha256=study.digest(__file__),
                review_amendment_sha256=study.review_revision(plan, audit['condition']))


def adopt_probes(plan):
    windows = {w['name']: w for w in plan['windows']}
    for path in sorted((OUT / 'capability_probes').glob('*.json')):
        item = read(path)
        if not item.get('schema_valid') or item.get('finish_reason') != 'stop':
            continue
        window = windows[item['window']]
        body, parent, mode = request(window, item['condition'], plan)
        if sha(body) != item['request_sha256']:
            continue
        saved = OUT / 'annotations' / item['condition'] / f'{window["name"]}.json'
        if saved.exists():
            continue
        audit = {key: value for key, value in item.items() if key not in ['response', 'events']}
        audit.update(mode=mode, parent_annotation_sha256=parent, reused_probe=str(path),
                     reused_probe_sha256=study.digest(path))
        study.write(saved, canonical(audit, item['response'], window, plan))


def annotate(window, condition, plan, budget, secret):
    saved = OUT / 'annotations' / condition / f'{window["name"]}.json'
    body, parent, mode = request(window, condition, plan)
    if saved.exists():
        item = read(saved)
        if (item['manifest_sha256'] != study.digest(OUT / 'manifest.json') or not item['mode_verified']
                or item['mode'] != mode or item['parent_annotation_sha256'] != parent):
            raise ValueError('Cached annotation provenance mismatch')
        study.parsed_events(item['raw_annotation'], window)
        return
    folder = OUT / 'attempts' / condition / window['name']
    previous = list(folder.glob('*.json'))
    if len(previous) >= 3:
        raise RuntimeError('Three attempts already used; inspect this window before retrying')
    for attempt in range(len(previous), 3):
        body, parent, mode = request(window, condition, plan, attempt)
        text_bytes, images = 0, 0
        for message in body['messages']:
            if isinstance(message['content'], str):
                text_bytes += len(message['content'].encode())
            else:
                text_bytes += sum(len(p['text'].encode()) for p in message['content'] if p['type'] == 'text')
                images += sum(p['type'] == 'image_url' for p in message['content'])
        input_bound = text_bytes + images * 16384 + 2048
        maximum = (input_bound * .30 + body['max_tokens'] * 1.20) / 1e6
        stamp = str(time.time_ns())
        budget.reserve(stamp, maximum)
        started = time.time()
        audit = dict(condition=condition, mode=mode, window=window['name'], attempt=attempt,
                     started_unix=started, max_tokens=body['max_tokens'], input_token_bound=input_bound,
                     reserved_usd=maximum, request_sha256=sha(body), parent_annotation_sha256=parent,
                     requested_model=MODEL, reasoning_effort=body['reasoning_effort'])
        path = folder / f'{stamp}.json'
        cost, state, failure, retry_after = None, 'unknown_charge', None, 2 ** attempt
        try:
            study.write(path, dict(audit, state='submitted'))
            response = requests.post(probe.URL, json=body, timeout=(20, 900), allow_redirects=False,
                                     headers={'Authorization': 'Bearer ' + secret, 'User-Agent': 'birdsong-research-pilot/1.0'})
            audit.update(http_status=response.status_code, elapsed_seconds=time.time() - started)
            raw = response.json()
            if secret in json.dumps(raw):
                raise RuntimeError('Credential unexpectedly echoed; response withheld')
            audit['response'] = raw
            usage = raw.get('usage') or {}
            if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ['prompt_tokens', 'completion_tokens']):
                audit['usage'] = usage
                cost = charge(usage, started, time.time())
                state = 'accounted'
                if usage['prompt_tokens'] > input_bound or usage['completion_tokens'] > body['max_tokens']:
                    raise RuntimeError('Token reservation assumptions exceeded')
            if response.status_code in (400, 401, 402, 403, 404):
                raise RuntimeError('API rejected request: HTTP ' + str(response.status_code))
            if response.status_code == 429:
                delay = response.headers.get('Retry-After', '10')
                try:
                    retry_after = max(float(delay), 10)
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    retry_after = max(10, parsedate_to_datetime(delay).timestamp() - time.time())
                with budget.condition:
                    budget.not_before = max(budget.not_before, time.time() + retry_after)
                raise ValueError('Throttled; honor Retry-After before retrying')
            response.raise_for_status()
            if cost is None:
                raise RuntimeError('Successful response omitted usage; cannot safely budget')
            item = canonical(audit, raw, window, plan)
            audit.update(accepted=True, reasoning_characters=item['reasoning_characters'])
            item.pop('response', None)
            study.write(saved, item)
        except (ValueError, KeyError, requests.RequestException) as error:
            failure = error
            audit.update(accepted=False, error=type(error).__name__ + ': ' + str(error).replace(secret, '[redacted]'))
        except BaseException as error:
            failure = error
            STOP.set()
            audit.update(accepted=False, error=type(error).__name__ + ': ' + str(error).replace(secret, '[redacted]'))
        finally:
            audit.update(elapsed_seconds=time.time() - started, accounted_upper_usd=cost, billing_state=state)
            study.write(path, audit)
            budget.settle(stamp, cost, state)
        if failure is None:
            return
        if STOP.is_set() or attempt == 2:
            raise failure
        if retry_after > 900:
            raise RuntimeError('Long provider cooldown; paused instead of retrying sooner')
        if STOP.wait(retry_after):
            raise InterruptedError('Paused during retry backoff')


def progress(budget, **extra):
    counts = {c: sum(1 for _ in (OUT / 'annotations' / c).glob('*.json')) for c in CONDITIONS}
    study.write(OUT / 'status.json', dict(updated_unix=time.time(), coverage=counts, total_per_condition=250,
                                        **budget.totals(), **extra))


def batch(tasks, workers, name, plan, budget, secret):
    for path, expected in plan['protected'].items():
        if study.digest(path) != expected:
            raise ValueError('Frozen input or runner changed: ' + path)
    print(f'{name}: {len(tasks)} tasks, {workers} workers', flush=True)
    progress(budget, state='annotating', stage=name, concurrency=workers)
    iterator, completed = iter(tasks), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = set()
        def submit():
            task = next(iterator, None)
            if task is not None and not STOP.is_set():
                active.add(pool.submit(annotate, *task, plan, budget, secret))
        for _ in range(workers):
            submit()
        try:
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    active.remove(future)
                    future.result()
                    completed += 1
                    progress(budget, state='annotating', stage=name, concurrency=workers)
                    print(f'{name}: {completed}/{len(tasks)}; upper cost ${budget.totals()["accounted_upper_usd"]:.3f}', flush=True)
                    submit()
        except BaseException:
            STOP.set()
            raise
    if completed != len(tasks):
        raise InterruptedError('Paused with work remaining')


def comparison(plan):
    rows = ['# Matched 250-window teacher pilot', '',
            '| Condition | Qwen pixel AP | Flash Vision Exp pixel AP | Qwen 2D IoU | Flash Vision Exp 2D IoU |',
            '| --- | ---: | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        path = OUT / 'scores' / f'{condition}.json'
        if not path.exists():
            continue
        other = read(SOURCE / 'scores' / f'{condition}.json')['evaluation']
        ours = study.score(plan, condition)['evaluation']
        rows.append(f'| {condition} | {other["ap"]:.4f} | {ours["ap"]:.4f} | {other["iou"]:.4f} | {ours["iou"]:.4f} |')
    rows += ['', 'Report: identical 150 Recording_1 windows; thresholds independently calibrated on the same 100 Recording_2–4 windows.',
             'Axes frozen from Qwen for both reasoning chains. AP uses continuous scores without smoothing.',
             'Zen model is deepseek-v4-flash-vision-exp; V4.1 identity is not established. Provider generation controls differ.',
             'Pilot teacher results only, not full Powdermill or student evaluation.']
    (OUT / 'comparison.md').write_text('\n'.join(rows) + '\n')


def run(plan, budget, secret, warmup_only):
    adopt_probes(plan)
    warmup = [w for w in plan['windows'] if w['partition'] == 'calibration'][:4]
    for condition in CONDITIONS:
        batch([(w, condition) for w in warmup], 4, 'warmup_' + condition, plan, budget, secret)
    if warmup_only:
        progress(budget, state='warmup_complete')
        return
    for condition in CONDITIONS:
        tasks = [(w, condition) for w in plan['windows']
                 if not (OUT / 'annotations' / condition / f'{w["name"]}.json').exists()]
        batch(tasks[:16], 16, condition + '_ramp', plan, budget, secret)
        batch(tasks[16:], 32, condition, plan, budget, secret)
        if sum(1 for _ in (OUT / 'annotations' / condition).glob('*.json')) != 250:
            raise ValueError('Incomplete condition; refusing to score a selected subset')
        progress(budget, state='scoring', stage=condition)
        study.score(plan, condition)
        comparison(plan)
    progress(budget, state='complete', conditions=CONDITIONS)
    probe.KEY.unlink(missing_ok=True)  # Remove only this job's private temporary credential.
    print('All five conditions scored on identical windows.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--warmup-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        budget = Budget()
        if args.prepare:
            progress(budget, state='prepared')
            print(json.dumps(budget.totals()), flush=True)
            return
        if not probe.KEY.is_file():
            raise ValueError('Private runtime credential is missing')
        secret = probe.credential()
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda *_: STOP.set())
        try:
            run(plan, budget, secret, args.warmup_only)
        except BaseException as error:
            progress(budget, state='paused', error=str(error).replace(secret, '[redacted]'))
            raise


if __name__ == '__main__':
    main()
