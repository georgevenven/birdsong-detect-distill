#!/usr/bin/env python3
"""Merge current-model shards after verifying frozen coverage and calibration."""
import argparse
import json
from pathlib import Path

from birdsong_detect_distill.benchmark_data import digest
from birdsong_detect_distill.benchmark_metrics import summarize
from evaluate_current_models import atomic_json
from merge_external_shards import by_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--model', choices=['songmae','qwen_yolo','birdbox'], required=True)
    parser.add_argument('--dataset', choices=['wabad','hawaii','xcsl','nips4bplus'], required=True)
    parser.add_argument('--shards', type=int, required=True)
    args = parser.parse_args()
    target = args.results / f'{args.model}_{args.dataset}.json'
    if target.exists():
        raise ValueError('final result already exists')
    manifest = json.loads((args.results / 'manifest.json').read_text())
    calibration_path = args.results / 'calibration' / f'{args.model}.json'
    calibration = json.loads(calibration_path.read_text())
    paths = [args.results / 'parts' / f'{args.model}_{args.dataset}_{i}.json' for i in range(args.shards)]
    parts = [json.loads(path.read_text()) for path in paths]
    reference_spec = manifest['external'][args.dataset]
    if digest(reference_spec['reference_report']) != reference_spec['reference_report_sha256']:
        raise ValueError('frozen reference report changed')
    reference = json.loads(Path(reference_spec['reference_report']).read_text())
    for i,part in enumerate(parts):
        if (part['sharding'] != dict(index=i, shards=args.shards) or part['dataset'] != args.dataset
                or part['model'] != args.model or part['protocol'] != calibration['protocol']
                or part['threshold_indices'] != calibration['threshold_indices']
                or part['calibration_sha256'] != digest(calibration_path)
                or part['reference_report_sha256'] != reference_spec['reference_report_sha256']):
            raise ValueError('incompatible shard or calibration provenance')
    rows = sorted([r for part in parts for r in part['per_recording']['evaluation']],key=lambda r:r['name'])
    if [r['name'] for r in rows] != sorted(reference_spec['recordings']):
        raise ValueError('missing or duplicated external recordings')
    expected = {r['name']:r for r in reference['per_recording']['evaluation']}
    for row in rows:
        other = expected[row['name']]
        if row['seconds'] != other['seconds'] or row['group'] != other['group']:
            raise ValueError('reference intervals or site groups differ')
        for metric,value in row['area'].items():
            if [tp+fn for tp,fp,fn in value['counts']] != [tp+fn for tp,fp,fn in other['area'][metric]['counts']]:
                raise ValueError('reference masks differ')
    sources = {}
    for part in parts:
        for name,sha in part['source_audio_sha256'].items():
            if name in sources and sources[name] != sha:
                raise ValueError('inconsistent source audio')
            sources[name] = sha
    if sources != reference['source_audio_sha256']:
        raise ValueError('audio differs from released-model evaluation')
    report = {k:v for k,v in parts[0].items() if k not in ['per_site','site_macro','sharding']}
    report.update(diagnostic_only=False, segments=len(rows), evaluated_seconds=sum(r['seconds'] for r in rows),
        summary=summarize(rows,calibration['threshold_indices']), per_recording={'evaluation':rows},
        source_audio_sha256=sources, reference_annotation_policy=reference_spec['reference_annotation_policy'],
        merged_shards={str(p):digest(p) for p in paths}, merge_code_sha256=digest(__file__),
        coverage_verification='same primary recordings, source hashes, intervals, site groups and reference positives as released baseline')
    if args.dataset in ['wabad','hawaii']:
        report['per_site'],report['site_macro'] = by_site(rows,calibration['threshold_indices'])
    atomic_json(target,report)
    print(json.dumps(dict(model=args.model,dataset=args.dataset,segments=len(rows),
        summary=report['site_macro'] if args.dataset=='wabad' else report['summary']),indent=2))


if __name__ == '__main__':
    main()
