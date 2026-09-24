#!/usr/bin/env python3
"""Validate a frozen study condition, then reuse the existing Powdermill scorer."""
import argparse
import json
from pathlib import Path

import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from evaluate_current_backbones import score_checkpoint
from prepare_detector_study import OUT, RAW, verify


def evaluate(spec_path):
    plan = json.loads((OUT / 'manifest.json').read_text())
    verify(plan)
    spec = json.loads(spec_path.read_text())
    data, val = plan['datasets'][spec['dataset']], plan['datasets']['validation']
    checkpoint = Path(spec['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = dict(backbone_id=plan['backbone'], backbone_revision=plan['revision'],
        hidden=spec['width'], head_layers=spec['layers'], seed=spec['seed'], dropout=.1,
        target_smoothing=False, tv_weight=0, ignore_uncertain=False, confidence_targets=False,
        height=4, width=1000, training_config=plan['training_config'],
        annotations_sha256=data['sha256'], validation_annotations_sha256=val['sha256'],
        training_recording_ids=data['recording_ids'], validation_recording_ids=val['recording_ids'])
    for key, value in expected.items():
        if saved[key] != value:
            raise ValueError(f'unmatched checkpoint configuration: {key}')
    metrics = saved['metrics']
    for key, value in dict(train_timebins=data['seconds'] * 200, validation_timebins=160000,
            train_windows=data['windows'], validation_windows=val['windows'], epochs=5,
            checkpoint_selection='minimum_recording_disjoint_validation_loss',
            threshold_source='external_calibration_required').items():
        if metrics[key] != value:
            raise ValueError(f'unmatched training budget/selection: {key}')
    if len(saved['training_history']) != 5 or metrics['selected_epoch'] != min(saved['training_history'], key=lambda r:r['validation_loss'])['epoch']:
        raise ValueError('checkpoint was not selected by XC validation loss within five epochs')
    anchor = json.loads(Path(plan['anchor']).read_text())
    reference = anchor['protocol']
    for filename, key in [('wav_Files.zip','audio_sha256'), ('annotation_Files.zip','references_sha256')]:
        if digest(RAW / 'powdermill' / filename) != reference[key]:
            raise ValueError('Powdermill audio or references changed')
    protocol = {key:reference[key] for key in ['partitions', 'coordinate_space', 'mask_rule',
        'threshold_selection', 'threshold_grid', 'aggregation', 'ap_method']}
    protocol.update(anchor=plan['anchor'], anchor_sha256=digest(plan['anchor']),
        checkpoint={**{k:v for k,v in saved.items() if k != 'head'}, 'checkpoint':str(checkpoint),
            'checkpoint_sha256':digest(checkpoint), 'trainable_parameters':sum(v.numel() for v in saved['head'].values())},
        student_inference={k:v for k,v in reference['student_inference'].items() if k != 'visible_devices'},
        code_sha256=plan['code_sha256'], study_manifest_sha256=digest(OUT / 'manifest.json'),
        condition=spec, condition_sha256=digest(spec_path),
        prediction_cache=dict(policy='samples', retained_segments=[reference['partitions']['calibration'][0],
            *reference['partitions']['evaluation'][:2]]))
    out = Path(spec['report']).parent
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('existing evaluation protocol changed')
    if not path.exists():
        write_json(path, protocol)
    if not Path(spec['report']).exists():
        score_checkpoint(checkpoint, saved, out, checkpoint.parent / 'predictions' / checkpoint.stem,
            RAW, reference, anchor, 'samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition', type=Path)
    evaluate(parser.parse_args().condition)
