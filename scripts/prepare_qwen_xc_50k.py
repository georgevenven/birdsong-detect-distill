#!/usr/bin/env python3
"""Freeze a species/geography-balanced XC sample and the successful Powdermill protocol."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from qwen_prompt_study import ROOT, context, digest, write

SPEC = ROOT / 'data/xcl'
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
POWDER = ROOT / 'results/qwen_teacher_powdermill/prompt_full_2026-09-11'
XC = re.compile(r'XC\d+', re.I)


def identifiers(values):
    return {m.group().upper() for value in values for m in XC.finditer(str(value))}


def exclusions():
    inventories, protected = {}, {}
    for dataset in ['xcaj', 'powdermill', 'wabad']:
        members = []
        archives = sorted((RAW / dataset).glob('*.zip'))
        if not archives:
            raise ValueError('missing evaluation archives: ' + dataset)
        for path in archives:
            with zipfile.ZipFile(path) as archive:
                members.extend((str(path), i.filename, i.CRC, i.file_size) for i in archive.infolist())
        inventories[dataset] = dict(xc_ids=sorted(identifiers(r[1] for r in members)),
            archive_count=len(archives), member_count=len(members),
            member_inventory_sha256=hashlib.sha256(json.dumps(members).encode()).hexdigest())
    for dataset, folder in [('hawaii', ROOT/'data/hawaii/zenodo'), ('nips4bplus', ROOT/'data/nips4bplus')]:
        audio = sorted((folder/'audio').glob('*'))
        if not audio:
            raise ValueError('missing evaluation audio: ' + dataset)
        inventories[dataset] = dict(xc_ids=sorted(identifiers(p.name for p in audio)),
            audio_count=len(audio), filename_sha256=hashlib.sha256(json.dumps([p.name for p in audio]).encode()).hexdigest())
    validation_index = Path('/home/george-vengrovski/Documents/SongMAE/data/XCL_val/shards/index.tsv')
    with validation_index.open() as stream:
        validation = {r['name'].upper() for r in csv.DictReader(stream, delimiter='\t')}
    split = ROOT/'data/annotations/xcl/stage_pipeline_10000train_800val_2026-09-09/manifest.json'
    detector_validation = set(json.loads(split.read_text())['partitions']['validation']['recording_ids'])
    inventories['xcl_validation'] = dict(xc_ids=sorted(validation))
    inventories['detector_validation'] = dict(xc_ids=sorted(detector_validation))
    for path in [validation_index, split, ROOT/'data/hawaii/zenodo/manifest.json',
                 ROOT/'data/nips4bplus/source/README.txt', RAW/'wabad/Pooled annotations.csv']:
        protected[str(path)] = digest(path)
    excluded = set().union(*(set(v['xc_ids']) for v in inventories.values()))
    return excluded, inventories, protected


def geography(record):
    lat, lon = record.get('lat'), record.get('long')
    if (not isinstance(lat, (int, float)) or not isinstance(lon, (int, float))
            or not math.isfinite(lat) or not math.isfinite(lon)
            or not -90 <= lat <= 90 or not -180 <= lon <= 180 or (lat == 0 and lon == 0)):
        return 'unknown', 'unknown'
    return f'{math.floor(lat)},{math.floor(lon)}', f'{lat:.2f},{lon:.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or args.seconds <= 0 or args.seconds % 5:
        raise ValueError('use a new output directory and a positive multiple of five seconds')
    excluded, inventory, protected = exclusions()
    source = ROOT/'data/annotations/xcl/XCL_train_annotations.json'
    metadata = {Path(r['recording']['filename']).stem.upper(): r['recording']
        for r in json.loads(source.read_text(),parse_constant=lambda _:None)['recordings']}
    with (SPEC/'shards/index.tsv').open() as stream:
        index = list(csv.DictReader(stream, delimiter='\t'))
    if len({r['name'] for r in index}) != len(index):
        raise ValueError('duplicate source recording IDs in the shard index')
    pool = defaultdict(list)
    for row in index:
        name = row['name'].upper()
        if name in excluded or int(row['end']) - int(row['start']) < 1001:
            continue  # Reserve the centered-STFT endpoint; never count padding as audio.
        record = metadata[name]
        if record['source'] != 'xenocanto':
            raise ValueError('unexpected source: ' + name)
        pool[record['ebird_code']].append(row)
    rng = random.Random(args.seed)
    for rows in pool.values():
        rng.shuffle(rows)
    cells, sites, recordists = Counter(), Counter(), Counter()
    windows, seen_specs = [], set()
    species = sorted(pool)
    target = args.seconds // 5
    duplicates = 0
    while len(windows) < target:
        rng.shuffle(species)
        progressed = False
        for code in species:
            while pool[code] and len(windows) < target:
                def priority(row):
                    record = metadata[row['name']]
                    cell, site = geography(record)
                    return cell == 'unknown', cells[cell], sites[site], recordists[record['recordist']]
                row = min(pool[code], key=priority)
                pool[code].remove(row)
                record = metadata[row['name']]
                start, end = int(row['start']), int(row['end'])
                offset = rng.randrange((end - start - 1) // 1000) * 1000
                tile = [row['name'], row['shard'], start, end, offset, offset + 1000]
                spec, _, _ = context(SPEC, tile, offset, offset + 1000)
                if spec.shape != (128, 1000) or not np.isfinite(spec).all():
                    raise ValueError('invalid source spectrogram: ' + row['name'])
                fingerprint = hashlib.sha256(spec.tobytes(order='C')).hexdigest()
                if fingerprint in seen_specs:
                    duplicates += 1
                    continue
                seen_specs.add(fingerprint)
                cell, site = geography(record)
                cells[cell] += 1; sites[site] += 1; recordists[record['recordist']] += 1
                windows.append(dict(name=f'{row["name"]}_{offset}_{offset+1000}', tile=tile,
                    seed=args.seed + 31 * len(windows), spectrogram_sha256=fingerprint,
                    metadata=record, geographic_cell_1_degree=cell, coordinate_site_0_01_degree=site))
                progressed = True
                if len(windows) % 500 == 0:
                    print(f'Validated {len(windows)}/{target} diverse five-second windows', flush=True)
                break
            if len(windows) == target:
                break
        if not progressed:
            raise ValueError('not enough distinct eligible recordings')
    chosen_ids = {w['tile'][0] for w in windows}
    assert len(chosen_ids) == target and not chosen_ids & excluded
    counts = Counter(w['metadata']['ebird_code'] for w in windows)
    selection = dict(target_seconds=args.seconds, windows=target, recordings=len(chosen_ids),
        species=len(counts), species_window_counts=dict(sorted(counts.items())),
        geographic_cells_1_degree=len(cells)-('unknown' in cells), coordinate_sites_0_01_degree=len(sites)-('unknown' in sites),
        unknown_coordinate_windows=cells['unknown'], recordists=len(recordists),
        geographic_cell_window_counts=dict(cells), max_windows_per_species=max(counts.values()),
        duplicate_selected_spectrograms_rejected=duplicates,
        method='Seeded species round-robin; one random aligned 5 s window per distinct recording. Within species prefer known coordinates, least represented 1-degree cell, 0.01-degree coordinate site, then recordist; seeded random ties. No activity/quality/prediction filtering.',
        prior_selection='Historical continuation uniformly shuffled recording IDs, one random aligned window each; it did not explicitly balance species or geography.',
        exclusions={k:dict(**v, selected_overlap=sorted(chosen_ids & set(v['xc_ids']))) for k,v in inventory.items()},
        caveat='Source-ID disjointness, not acoustic fingerprint proof: XC-AJ IDs (including all original splits) and validation IDs are excluded. Other corpora use independent non-XC filename namespaces. Reuploads under different IDs or extracted/reencoded duplicate audio cannot be ruled out from the available XCL spectrograms/metadata. Species/location counts describe recording metadata, not necessarily vocalizations inside each sampled window.')
    powder = json.loads((POWDER/'manifest.json').read_text())
    keys = ['system_prompt','self_review_prompt','schema','canvas','plot_rectangle_pixels','label',
        'numbered_review','review_amendment_sha256','temperature','top_p','top_k','reasoning','no_reasoning','context_per_slot']
    for path in [source, SPEC/'shards/index.tsv', SPEC/'audio_params.json', POWDER/'manifest.json',
            ROOT/'results/qwen_teacher_powdermill/prompt_250_2026-09-11/review_amendment.json',
            ROOT/'scripts/qwen_prompt_study.py', ROOT/'scripts/run_qwen_prompt_study.py',
            ROOT/'src/birdsong_detect_distill/qwen.py', ROOT/'src/birdsong_detect_distill/data.py',
            Path(__file__), ROOT/'scripts/run_qwen_xc_50k.py',
            Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
            Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')]:
        protected[str(path)] = digest(path)
    for shard in {w['tile'][1] for w in windows}:
        path = SPEC/'shards'/Path(shard).with_suffix('.txt')
        protected[str(path)] = digest(path)
    # The shared prompt/renderer/request code must be byte-identical to full Powdermill.
    for name in ['scripts/qwen_prompt_study.py','scripts/run_qwen_prompt_study.py',
                 'src/birdsong_detect_distill/qwen.py','src/birdsong_detect_distill/data.py']:
        assert protected[str(ROOT/name)] == powder['protected'][str(ROOT/name)]
    write(out/'selection.json', selection)
    protected[str(out/'selection.json')] = digest(out/'selection.json')
    backends = {}
    for role, old_role in [('twins','twins'),('v100','vector')]:
        source_backend = POWDER/'backfill'/old_role/'backend.json'
        old = json.loads(source_backend.read_text())
        backends[role] = {k:old[k] for k in ['url','workers','context_per_slot','remote_model_path','model_sha256',
            'mmproj_sha256','llama_cpp_commit','request_timeout_seconds']}
        backends[role].update(role=role, source_backend=str(source_backend), source_backend_sha256=digest(source_backend),
            physical_host='Lambda-Twins' if role=='twins' else 'george-server', gpu_count=2 if role=='twins' else 3)
        protected[str(source_backend)] = digest(source_backend)
    plan = {**{k:powder[k] for k in keys}, 'windows':windows, 'protected':protected, 'backends':backends,
        'conditions':['reasoning','self_review_1'], 'selected_image_format':'axes', 'target_seconds':args.seconds,
        'spec_dir':str(SPEC), 'source_powdermill_manifest_sha256':digest(POWDER/'manifest.json'),
        'execution':'Two calls per window: reasoning then one numbered self-review, each checkpointed separately. Shared whole-window locks; Twins and V100 only. Viewing is a network relay, never an inference backend.',
        'old_annotation_reuse':False, 'training_validation_note':'This is an annotation pool, not a new train/validation split; historical validation recordings remain excluded.',
        'spectrogram_note':'Same native 128-mel/5-ms frontend and fixed viridis/axes renderer. Existing XC int8-affine shards are dequantized with their sidecars; no resizing of frequency/time inputs, renormalization, context shift, or altered prompting.'}
    write(out/'manifest.json', plan)
    print(json.dumps({k:v for k,v in selection.items() if k not in ['exclusions','species_window_counts','geographic_cell_window_counts']}, indent=2))


if __name__ == '__main__':
    main()
