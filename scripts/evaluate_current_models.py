#!/usr/bin/env python3
"""Matched SongMAE/YOLO evaluation with resumable scores and bounded dense-map storage."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from birdsong_detect_distill import hawaii
from birdsong_detect_distill.baseline_models import Predictor, maps
from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import RATE, area_scores, calibrate, intervals_mask, summarize
from evaluate_baselines import INFERENCE_FILES
from evaluate_songmae_smoothing import smooth
from merge_external_shards import by_site


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    write_json(temporary, value)
    temporary.replace(path)


def check_training(kind, checkpoint, manifest):
    train, val = manifest['training'], manifest['validation']
    if kind == 'songmae':
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        actual = dict(train_sha=saved['annotations_sha256'], val_sha=saved['validation_annotations_sha256'],
            train_ids=saved['training_recording_ids'], val_ids=saved['validation_recording_ids'],
            train_seconds=saved['metrics']['train_timebins']/RATE, val_seconds=saved['metrics']['validation_timebins']/RATE)
        if saved['target_smoothing'] or saved['tv_weight'] != 0:
            raise ValueError('expected hard-target BCE SongMAE checkpoint')
    else:
        saved = json.loads(checkpoint.with_suffix('.json').read_text())
        d = saved['dataset']
        actual = dict(train_sha=d['annotations_sha256'], val_sha=d['validation_annotations_sha256'],
            train_ids=d['training_recordings'], val_ids=d['validation_recordings'],
            train_seconds=d['training_seconds'], val_seconds=d['validation_seconds'])
        if saved['sha256'] != digest(checkpoint) or d['train_all']:
            raise ValueError('expected validation-selected YOLO with a valid sidecar')
    expected = dict(train_sha=train['files']['self_review']['sha256'], val_sha=val['files']['self_review']['sha256'],
        train_ids=train['recording_ids'], val_ids=val['recording_ids'], train_seconds=10000, val_seconds=800)
    if actual != expected:
        raise ValueError('checkpoint does not use the matched training/validation split')
    return {k:v for k,v in saved.items() if k != 'head'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['songmae', 'qwen_yolo', 'birdbox'], required=True)
    parser.add_argument('--dataset', choices=['powdermill','wabad','hawaii','xcsl','nips4bplus'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--variant', choices=['yolo11n', 'yolo11l'], default='yolo11n')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if (args.model == 'birdbox') == (args.checkpoint is not None):
        raise ValueError('released BirdBox uses pinned variant weights; students require --checkpoint')
    if args.model != 'birdbox' and args.variant != 'yolo11n':
        raise ValueError('--variant applies only to released BirdBox')
    if args.out.exists() or not 0 <= args.shard_index < args.shards:
        raise ValueError('choose a new result and valid shard identity')
    if args.dataset == 'powdermill' and args.shards != 1:
        raise ValueError('calibration may not be sharded')
    if args.dataset != 'powdermill' and not args.calibration:
        raise ValueError('external evaluation requires frozen calibration')
    manifest = json.loads(args.manifest.read_text())
    if digest(manifest['split']) != manifest['split_sha256']:
        raise ValueError('XC split changed')
    saved = check_training(args.model, args.checkpoint, manifest) if args.model != 'birdbox' else None
    revision = saved['backbone_revision'] if args.model == 'songmae' else None
    predictor = Predictor(args.model, args.checkpoint, args.device, batch_size=8,
        confidence=.00001, max_det=10000, backbone_revision=revision, variant=args.variant)
    code = {p:digest(p) for p in INFERENCE_FILES}
    metadata = {**predictor.metadata, 'code_sha256':code, 'matched_training':saved}
    scoring = dict(code_sha256={p:digest(p) for p in [__file__, 'scripts/evaluate_songmae_smoothing.py',
        'src/birdsong_detect_distill/benchmark_metrics.py', 'scripts/merge_external_shards.py']},
        songmae_smoothing='Gaussian probability sigma=(2 mel bins,3 frames), reflect, truncate=4; before frequency max' if args.model == 'songmae' else None,
        binary_postprocessing='single calibrated threshold; no morphological cleanup',
        references='drop zero-area rectangles before rasterization; preserve frozen external exclusions',
        aggregation='WABAD site macro; other datasets recording macro; AP omits empty-positive recordings')
    protocol = dict(inference=metadata, scoring=scoring, manifest_sha256=digest(args.manifest))
    protocol_hash = fingerprint(protocol)
    args.cache.mkdir(parents=True, exist_ok=True)
    with (args.cache / 'metadata.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.cache / 'metadata.json'
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError('cache protocol changed')
        if not path.exists():
            atomic_json(path, protocol)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if calibration and (calibration['protocol'] != protocol or calibration['diagnostic_only']
            or calibration['dataset'] != 'powdermill'):
        raise ValueError('calibration model or protocol differs')
    inventory = {r['name']:r for r in (hawaii.recordings(args.root) if args.dataset == 'hawaii' else recordings(args.root, args.dataset))}
    reference = None
    if args.dataset == 'powdermill':
        partitions = manifest['powdermill']
    else:
        selected = manifest['external'][args.dataset]
        if digest(selected['reference_report']) != selected['reference_report_sha256']:
            raise ValueError('reference competitor result changed')
        reference = json.loads(Path(selected['reference_report']).read_text())
        partitions = {'evaluation':selected['recordings']}
    names = sorted(n for values in partitions.values() for n in values)
    if set(names) - inventory.keys() or len(names) != len(set(names)):
        raise ValueError('missing or duplicated recording coverage')
    examples = {names[0], names[len(names)//2], names[-1]}
    partitions = {part:sorted(values)[args.shard_index::args.shards] for part,values in partitions.items()}
    expected_rows = {r['name']:r for r in reference['per_recording']['evaluation']} if reference else {}
    stage, stage_rows = None, {}
    if args.dataset == 'powdermill' and args.model == 'songmae':
        stage_root = Path(manifest['stage_run'])
        if digest(stage_root / 'protocol.json') != manifest['stage_protocol_sha256']:
            raise ValueError('stage-run provenance changed')
        stage = json.loads((stage_root / 'protocol.json').read_text())
        if stage['checkpoints']['self_review']['checkpoint_sha256'] != metadata['checkpoint_sha256']:
            raise ValueError('stage cache uses a different student')
        if stage['references_sha256'] != digest(args.root / 'powdermill/annotation_Files.zip'):
            raise ValueError('Powdermill references changed')
        stage_report = json.loads((stage_root / 'comparison.json').read_text())
        stage_rows = {r['name']:r for rows in stage_report['per_recording']['student_self_review'].values() for r in rows}
    source_hashes, scored = {}, {part:[] for part in partitions}
    for partition, selected in partitions.items():
        for index, name in enumerate(selected, 1):
            record = inventory[name]
            audio_source = record.get('archive', record.get('path'))
            if audio_source not in source_hashes:
                source_hashes[audio_source] = digest(audio_source)
            audio_hash = source_hashes[audio_source]
            if record.get('expected_audio_sha256', audio_hash) != audio_hash:
                raise ValueError(f'audio differs from release: {name}')
            if reference and reference['source_audio_sha256'][audio_source] != audio_hash:
                raise ValueError(f'audio differs from released-model evaluation: {name}')
            signature = dict(audio_sha256=audio_hash, member=record.get('member'),
                reference_sha256=hashlib.sha256(record['events'].tobytes()).hexdigest(), ignored=record.get('ignored', []))
            stem = name.replace('/', '__')
            score_path = args.cache / 'scores' / args.dataset / f'{stem}.json'
            if score_path.exists():
                cached = json.loads(score_path.read_text())
                if cached['source'] != signature or cached['protocol_sha256'] != protocol_hash:
                    raise ValueError(f'cached score provenance changed: {name}')
                row = cached['row']
            else:
                if stage:
                    if stage['audio_sha256'] != audio_hash:
                        raise ValueError('stage audio source differs')
                    path = Path(stage['checkpoints']['self_review']['checkpoint']).parent / 'predictions' / f'{name}.npz'
                    audit = json.loads((stage_root / 'segments' / f'{name}.json').read_text())
                    if digest(path) != audit['prediction_sha256']:
                        raise ValueError(f'stage prediction changed: {name}')
                    with np.load(path) as archive:
                        prediction = dict(probability=archive['self_review'])
                    duration = stage_rows[name]['seconds']
                else:
                    audio, rate = read_audio(record)
                    duration = len(audio)/rate
                    audio, rate = model_audio(audio, rate, args.model)
                    prediction = predictor.predict(audio, rate)
                raw_hash = hashlib.sha256()
                for key,value in sorted(prediction.items()):
                    raw_hash.update(key.encode())
                    raw_hash.update(np.ascontiguousarray(value).tobytes())
                prediction_path = None
                if not stage and (args.model in ['qwen_yolo', 'birdbox'] or name in examples):
                    path = args.cache / 'predictions' / args.dataset / f'{stem}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, **prediction)
                    temporary.replace(path)
                    prediction_path = str(path)
                if args.model == 'songmae':
                    prediction = dict(probability=smooth(prediction['probability']))
                probability, score = maps(prediction, duration)
                events = record['events']
                if ((events[:,1] < events[:,0]) | (events[:,3] < events[:,2])).any():
                    raise ValueError(f'inverted reference bounds in retained recording: {name}')
                events = events[(events[:,1] > events[:,0]) & (events[:,3] > events[:,2])]
                intervals, ignored = [[0,duration]], record.get('ignored', [])
                row = dict(name=name, group=record['group'],
                    seconds=float(intervals_mask(intervals, len(score), ignored).sum()/RATE),
                    area=area_scores(probability, score, events, len(score), intervals, ignored, record.get('temporal_only',False)))
                if stage and row['area']['full'] != stage_rows[name]['area']['full']:
                    raise ValueError(f'smoothed student pixel scores do not reproduce stage evaluation: {name}')
                atomic_json(score_path, dict(protocol_sha256=protocol_hash, source=signature, duration=duration,
                    raw_prediction_sha256=raw_hash.hexdigest(), prediction_path=prediction_path,
                    prediction_file_sha256=digest(prediction_path) if prediction_path else None, row=row))
            if reference:
                expected = expected_rows[name]
                if row['seconds'] != expected['seconds']:
                    raise ValueError(f'scoring duration differs from competitors: {name}')
                for metric, value in row['area'].items():
                    if [tp+fn for tp,fp,fn in value['counts']] != [tp+fn for tp,fp,fn in expected['area'][metric]['counts']]:
                        raise ValueError(f'reference mask differs from competitors: {name}/{metric}')
            scored[partition].append(row)
            print(f'{args.model} {args.dataset} {partition} {index}/{len(selected)}: {name}', flush=True)
        if partition == 'calibration':
            thresholds = calibrate(scored['calibration'])
            atomic_json(args.out.with_suffix('.thresholds.json'), dict(protocol_sha256=protocol_hash,
                threshold_indices=thresholds, selection=manifest['threshold_selection'], recordings=selected))
            print(f'Thresholds frozen before reporting: {thresholds}', flush=True)
    if calibration:
        thresholds = calibration['threshold_indices']
    report = dict(dataset=args.dataset, model=args.model, protocol=protocol,
        diagnostic_only=args.shards>1, sharding=dict(index=args.shard_index, shards=args.shards),
        threshold_indices=thresholds, calibration_sha256=digest(args.calibration) if args.calibration else None,
        segments=len(scored['evaluation']), evaluated_seconds=sum(r['seconds'] for r in scored['evaluation']),
        summary=summarize(scored['evaluation'],thresholds), per_recording=scored, source_audio_sha256=source_hashes,
        reference_report_sha256=manifest['external'][args.dataset]['reference_report_sha256'] if reference else None)
    if args.dataset in ['wabad','hawaii']:
        report['per_site'], report['site_macro'] = by_site(scored['evaluation'], thresholds)
    atomic_json(args.out, report)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
