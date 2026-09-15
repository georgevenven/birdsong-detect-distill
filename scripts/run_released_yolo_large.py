#!/usr/bin/env python3
"""Add released YOLO11l to the frozen external comparison without replacing prior results."""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from birdsong_detect_distill.benchmark_data import digest
from birdsong_detect_distill.birdbox import model_path
from evaluate_baselines import INFERENCE_FILES
from evaluate_current_models import atomic_json

OUT = Path('results/yolo11l_external_2026-09-10')
OLD = Path('results/current_external_2026-09-09')
CACHE = Path('artifacts/yolo11l_external_2026-09-10')
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
GPUS = ['GPU-93189925-4747-5857-e01f-d44d77ad07a7', 'GPU-8d9ddeaf-7d5c-60bf-69f8-eb10172a2556']
SHARDS = 8


def run(arguments, log, gpu=None):
    env = {**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
        'OPENBLAS_NUM_THREADS':'2', 'HF_HUB_OFFLINE':'1', 'PYTHONUNBUFFERED':'1'}
    env['CUDA_VISIBLE_DEVICES'] = gpu or ''
    print('Running:', ' '.join(map(str, arguments)), flush=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable, '-u', *map(str, arguments)], env=env,
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    marker = Path('/home/george-vengrovski/.config/birdsong-detect-distill/qwen.paused')
    if not marker.exists():
        raise ValueError('Qwen must remain manually paused')
    for unit in ['songmae-qwen.service', 'birdsong-qwen-xc-continuation.service']:
        pid = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True).strip()
        if pid != '0':
            raise ValueError('Qwen is active; do not compete for its GPUs')
    if shutil.disk_usage('.').free < 3 * 1024**3:
        raise ValueError('need at least 3 GiB free before starting')
    for folder in ['logs', 'parts', 'calibration']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    if digest(manifest_path) != digest(OLD / 'manifest.json'):
        raise ValueError('evaluation manifest differs from completed comparison')
    checkpoint = model_path('yolo11l', Path('models'))
    sources = json.loads((OLD / 'table_sources.json').read_text())['reports_sha256']
    for name, sha in sources.items():
        if digest(name) != sha:
            raise ValueError(f'prior result changed: {name}')
    protected = {**sources, **{str(p):digest(p) for p in OLD.iterdir() if p.is_file()},
        **{str(p):digest(p) for p in (OLD / 'calibration').glob('*.json')}}
    code = set(INFERENCE_FILES + [__file__, 'scripts/evaluate_current_models.py',
        'scripts/merge_current_external.py', 'scripts/summarize_current_external.py',
        'scripts/merge_external_shards.py', 'src/birdsong_detect_distill/benchmark_metrics.py',
        'src/birdsong_detect_distill/hawaii.py', 'src/birdsong_detect_distill/evaluation.py',
        'scripts/evaluate_songmae_smoothing.py'])
    plan = dict(model='released_yolo11l', checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        manifest_sha256=digest(manifest_path), protected_files=protected,
        code_sha256={str(p):digest(p) for p in sorted(code)}, shards=SHARDS, batch_size=8,
        scope='released weights only; no training, SongMAE reruns or Qwen restart')
    frozen = OUT / 'run_manifest.json'
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError('frozen run inputs, code or historical results changed')
    if not frozen.exists():
        atomic_json(frozen, plan)
    common = ['scripts/evaluate_current_models.py', '--model', 'birdbox', '--variant', 'yolo11l',
        '--manifest', manifest_path, '--cache', CACHE]
    calibration = OUT / 'calibration/birdbox.json'
    if not calibration.exists():
        run([*common, '--dataset', 'powdermill', '--root', RAW, '--out', calibration],
            OUT / 'logs/calibration.log', GPUS[0])
    for dataset in ['nips4bplus', 'xcsl', 'wabad', 'hawaii']:
        if (OUT / f'birdbox_{dataset}.json').exists():
            continue
        root = Path('data/hawaii/zenodo') if dataset == 'hawaii' else Path('data/nips4bplus') if dataset == 'nips4bplus' else RAW
        with ThreadPoolExecutor(max_workers=SHARDS) as pool:
            futures = []
            for index in range(SHARDS):
                part = OUT / f'parts/birdbox_{dataset}_{index}.json'
                if part.exists():
                    continue
                arguments = [*common, '--dataset', dataset, '--root', root, '--out', part,
                    '--calibration', calibration, '--shards', str(SHARDS), '--shard-index', str(index)]
                futures.append(pool.submit(run, arguments, OUT / f'logs/{dataset}_{index}.log', GPUS[index % len(GPUS)]))
            for future in futures:
                future.result()
        run(['scripts/merge_current_external.py', '--model', 'birdbox', '--dataset', dataset,
            '--results', OUT, '--shards', str(SHARDS)], OUT / f'logs/merge_{dataset}.log')
    for name, sha in {**protected, **plan['code_sha256'], str(checkpoint):plan['checkpoint_sha256']}.items():
        if digest(name) != sha:
            raise ValueError(f'protected file changed during evaluation: {name}')
    if not (OUT / 'tables/tables.md').exists():
        run(['scripts/summarize_current_external.py', '--results', OLD, '--released-large', OUT,
            '--out', OUT / 'tables'], OUT / 'logs/tables.log')
    print(f'Complete: {OUT}/tables/tables.md. Qwen remains paused.', flush=True)


if __name__ == '__main__':
    main()
