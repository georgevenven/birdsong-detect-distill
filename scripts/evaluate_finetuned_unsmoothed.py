#!/usr/bin/env python3
"""Ablate smoothing only; cross-calibrate unchanged fine-tuned checkpoints on Powdermill."""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_full_encoder_backbones_snapshot as trained
from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, THRESHOLDS, area, calibrate, summarize
from birdsong_detect_distill.full_encoder_linear import load_heads
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import songmae_probabilities
from evaluate_songmae_smoothing import truth_and_coverage
from prepare_detector_study import verify
from summarize_powdermill_loro import checked_rows, evaluate, group

OUT = trained.OUT.parent / 'unsmoothed_full_encoder_3205s_2026-09-14'
ARTIFACTS = trained.ARTIFACTS.parent / OUT.name


def run(size):
    out, cache = OUT / size, ARTIFACTS / size
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    lock = (out / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    study = json.loads((trained.OUT / 'manifest.json').read_text())
    verify(study)
    source_dir = trained.OUT / 'runs' / size
    report_path = source_dir / 'comparison.json'
    previous = json.loads(report_path.read_text())
    old_rows = {r['name']: r for r in checked_rows(previous)}
    baseline = json.loads((source_dir / 'loro.json').read_text())
    protocol = previous['protocol']
    checkpoint = Path(protocol['checkpoint_path'])
    if digest(checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    anchor = json.loads(Path(protocol['anchor']).read_text())['protocol']
    raw_root = trained.original.RAW
    for name, key in [('wav_Files.zip', 'audio_sha256'), ('annotation_Files.zip', 'references_sha256')]:
        if digest(raw_root / 'powdermill' / name) != anchor[key]:
            raise ValueError('Powdermill source changed')
    protected = [Path(__file__), report_path, source_dir / 'loro.json', checkpoint,
        *sorted((source_dir / 'segments').glob('*.json'))]
    plan = dict(size=size, training_seconds=study['datasets']['train']['seconds'],
        validation_seconds=study['datasets']['validation']['seconds'], seed=study['seed'],
        checkpoint=str(checkpoint), checkpoint_sha256=protocol['checkpoint_sha256'],
        inference='Unchanged checkpoint, native frontend, 5 s windows / 2.5 s stride, maximum overlap; no smoothing or morphology.',
        prediction_verification='Require every raw float32 prediction array to match its original saved SHA256 bitwise.',
        calibration='Leave one entire original recording out; maximize mean segment IoU on the other three; independent raw thresholds.',
        threshold_grid=THRESHOLDS.tolist(), tie_policy='lowest threshold',
        aggregation='Mean pixel AP over 76 positive segments; mean IoU over all 77; pooled precision/recall.',
        caveat='Single seed; exploratory Powdermill development analysis, not independent testing.',
        protected_files={str(p): digest(p) for p in protected}, code_sha256=study['code_sha256'])
    trained.frozen(out / 'manifest.json', plan)
    verify(plan)
    if (out / 'comparison.json').exists():
        print(size, 'already complete', flush=True)
        return
    trained.progress(out, 'scoring', completed=0, total=77)
    inventory = {r['name']: r for r in recordings(raw_root, 'powdermill')}
    if inventory.keys() != old_rows.keys():
        raise ValueError('Powdermill inventory changed')
    device = torch.device('cuda:0')
    backbone, heads = None, None
    rows, started = [], time.time()
    manifest_sha = digest(out / 'manifest.json')
    for index, name in enumerate(sorted(inventory), 1):
        result_path = out / 'segments' / f'{name}.json'
        path = cache / f'{name}.npz'
        original = json.loads((source_dir / 'segments' / f'{name}.json').read_text())
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['manifest_sha256'] != manifest_sha or digest(path) != result['prediction_sha256']:
                raise ValueError('changed resumed result')
        else:
            record = inventory[name]
            source = dict(audio_sha256=anchor['audio_sha256'], member=record['member'],
                reference_sha256=hashlib.sha256(record['events'].tobytes()).hexdigest())
            if source != original['source']:
                raise ValueError('source identity changed')
            retained = trained.ARTIFACTS / size / 'predictions' / f'{name}.npz'
            if retained.exists():
                if digest(retained) != original['prediction_sha256']:
                    raise ValueError('original retained cache changed')
                with np.load(retained) as saved:
                    probability = saved['probability']
            else:
                if backbone is None:
                    if torch.cuda.mem_get_info(0)[0] < 20 * 1024**3:
                        raise ValueError('GPU occupied; preserve existing work')
                    torch.manual_seed(0)
                    torch.set_float32_matmul_precision('high')
                    metadata = protocol['checkpoint']
                    backbone = load_backbone(metadata['backbone_id'], device, metadata['backbone_revision'])
                    heads = load_heads([checkpoint], backbone, device)
                audio, rate = read_audio(record)
                if len(audio) / rate != old_rows[name]['seconds']:
                    raise ValueError('source duration changed')
                audio, _ = model_audio(audio, rate, 'songmae')
                probability = songmae_probabilities(backbone, heads, audio, device)[checkpoint.stem]
            array_sha = hashlib.sha256(probability.tobytes()).hexdigest()
            if array_sha != original['prediction_array_sha256']:
                raise ValueError(f'raw prediction does not exactly reproduce original inference: {name}')
            seconds = old_rows[name]['seconds']
            width = int(np.ceil(seconds * RATE))
            truth, kept = truth_and_coverage(record, width, [[0, seconds]])
            if not kept.all() or probability.dtype != np.float32 or probability.shape != (128, width + 1):
                raise ValueError('unexpected scoring coverage or native grid')
            score = dict(name=name, group=record['group'], seconds=seconds,
                area={'full': area(probability[:, :width], truth)})
            if score['area']['full']['counts'][0] != old_rows[name]['area']['full']['counts'][0]:
                raise ValueError('reference area changed')
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, probability=probability)
            temporary.replace(path)
            result = dict(manifest_sha256=manifest_sha, source=source, score=score,
                prediction_sha256=digest(path), prediction_array_sha256=array_sha)
            write_json(result_path, result)
        rows.append(result['score'])
        trained.progress(out, 'scoring', completed=index, total=77, elapsed_seconds=time.time() - started)
        print(f'{size} {index}/77 {name}: raw AP={rows[-1]["area"]["full"]["ap"]}', flush=True)
    partitions = {key: [r for r in rows if (group(r) == 'Recording_1') == (key == 'evaluation')]
        for key in ['calibration', 'evaluation']}
    threshold = calibrate(partitions['calibration'])
    raw_report = dict(per_recording=partitions, summary=summarize(partitions['evaluation'], threshold)['full'])
    rows = checked_rows(raw_report)
    spec = dict(id=size, family='full_encoder_linear_unsmoothed', label=f'SongMAE-{size} without smoothing',
        seed=0, threshold_floor_index=0, path=out / 'scores.json')
    loro = evaluate(spec, raw_report, rows)
    write_json(out / 'scores.json', raw_report)
    write_json(out / 'loro.json', loro)
    metrics = ['ap', 'iou', 'pooled_precision', 'pooled_recall']
    comparison = dict(manifest_sha256=manifest_sha, size=size, raw=loro['summary'], smoothed=baseline['summary'],
        raw_minus_smoothed={k: loro['summary'][k] - baseline['summary'][k] for k in metrics},
        raw_thresholds={f['held_out']: f['threshold'] for f in loro['folds']},
        smoothed_thresholds={f['held_out']: f['threshold'] for f in baseline['folds']},
        recording1_raw=raw_report['summary'], recording1_smoothed=previous['summary'],
        raw_predictions_reproduce_original='bitwise SHA256 match for all 77 segments')
    verify(plan)
    write_json(out / 'comparison.json', comparison)
    trained.progress(out, 'complete', completed=77, total=77)
    print(json.dumps(comparison, indent=2), flush=True)
    del backbone, heads
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', choices=['micro', 'base', 'large'], required=True)
    args = parser.parse_args()
    for size in args.sizes:
        try:
            run(size)
        except Exception as error:
            trained.progress(OUT / size, 'failed', error=repr(error))
            raise
