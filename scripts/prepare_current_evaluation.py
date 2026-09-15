#!/usr/bin/env python3
"""Pin the new student split while preserving the completed competitors' external coverage."""
import argparse
import json
import re
from pathlib import Path

from birdsong_detect_distill.benchmark_data import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('choose a new manifest path')
    split_path = Path('data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09/manifest.json')
    old_path = Path('results/baselines_matched_2026-09-07/manifest.json')
    stage_path = Path('results/qwen_teacher_powdermill/stage_pipeline_10000train_800val_2026-09-09')
    split, old = [json.loads(p.read_text()) for p in (split_path, old_path)]
    stage = json.loads((stage_path / 'protocol.json').read_text())
    train, val = split['partitions']['train'], split['partitions']['validation']
    if train['seconds'] != 10000 or val['seconds'] != 800:
        raise ValueError('unexpected training/validation budget')
    if set(train['recording_ids']) & set(val['recording_ids']):
        raise ValueError('XC training/validation overlap')
    external_ids = {re.search(r'XC\d+', n, re.I).group().upper() for n in old['xcsl']['evaluation']}
    if external_ids & set(train['recording_ids'] + val['recording_ids']):
        raise ValueError('XC-AJ overlap with training or checkpoint selection')
    if old['powdermill']['calibration'] != stage['partitions']['calibration']:
        raise ValueError('Powdermill calibration split changed')
    folder = Path('results/competitors_external_2026-09-08')
    sources = json.loads((folder / 'table_sources.json').read_text())
    external = {}
    for dataset, count in [('wabad', 4264), ('hawaii', 635), ('xcsl', 288), ('nips4bplus', 674)]:
        path = folder / f'birdbox_{dataset}.json'
        report = json.loads(path.read_text())
        if digest(path) != sources['reports_sha256'][path.name] or report['segments'] != count:
            raise ValueError(f'changed reference competitor report: {dataset}')
        external[dataset] = dict(reference_report=str(path), reference_report_sha256=digest(path),
            recordings=[r['name'] for r in report['per_recording']['evaluation']],
            reference_annotation_policy=report['reference_annotation_policy'])
    write_json(args.out, dict(split=str(split_path), split_sha256=digest(split_path),
        source_manifest=str(old_path), source_manifest_sha256=digest(old_path),
        training=train, validation=val, teacher_variant='self_review',
        powdermill=stage['partitions'], stage_run=str(stage_path),
        stage_protocol_sha256=digest(stage_path / 'protocol.json'), external=external,
        metrics='Exact pixel/frame AP and separately calibrated area/temporal IoU on the existing 5-ms grid',
        threshold_selection='maximum mean segment IoU on Powdermill Recordings 2–4 only',
        caveat=old['caveat']))
    print(f'Frozen 10000s/800s split; unchanged external counts: { {k: len(v["recordings"]) for k,v in external.items()} }')


if __name__ == '__main__':
    main()
