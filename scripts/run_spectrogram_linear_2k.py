#!/usr/bin/env python3
"""Train a raw-spectrogram baseline; evaluate full Powdermill and cross-calibrate."""
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import train as trainer
from birdsong_detect_distill.benchmark_data import digest, write_json, recordings, read_audio, model_audio
from birdsong_detect_distill.benchmark_metrics import area, calibrate, summarize
from birdsong_detect_distill.data import PixelWindows, read_rows
from birdsong_detect_distill.spectrogram_linear import ARCHITECTURE, SpectrogramLinear, frontend
from evaluate_2d import songmae_probability
from evaluate_songmae_smoothing import smooth, truth_and_coverage
from prepare_detector_study import verify
from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate

ROOT = Path(__file__).resolve().parents[1]
RUN = 'raw_spectrogram_linear_2000train_800val_2026-09-14'
OUT = ROOT / 'results/qwen_teacher_powdermill' / RUN
ARTIFACTS = Path('/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments') / RUN
PARENT = ROOT / 'results/qwen_teacher_powdermill/new_teacher_2000train_800val_2026-09-14'
CONTROL = ROOT / 'results/qwen_teacher_powdermill/pointwise_bare_linear_2000train_800val_2026-09-14/runs/bare_linear_d0_new2k_s0/comparison.json'
LORO = ROOT / 'results/qwen_teacher_powdermill/leave_one_recording_out_2026-09-14/linear_s0.json'
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
CHECKPOINT = ARTIFACTS / 'spectrogram_linear_s0.pt'


def progress(state, **values):
    write_json(OUT / 'status.json', dict(state=state, updated_unix=time.time(), **values))


def prepare():
    parent = json.loads((PARENT / 'manifest.json').read_text())
    verify(parent)
    control = json.loads(CONTROL.read_text())
    saved = control['protocol']['checkpoint']
    config = frontend(saved['backbone_id'], saved['backbone_revision']).config
    protected = {**parent['protected_files'], **parent['code_sha256']}
    for p in [Path(__file__), CONTROL, LORO, PARENT / 'manifest.json',
              ROOT / 'src/birdsong_detect_distill/spectrogram_linear.py', ROOT / 'scripts/summarize_powdermill_loro.py']:
        protected[str(p)] = digest(p)
    plan = dict(architecture=ARCHITECTURE, parameters=16512, seed=0, epochs=5,
        learning_rate=.001, weight_decay=.0001, batch_size=16, validation_batch_size=4,
        frontend_reference=saved['backbone_id'], frontend_revision=saved['backbone_revision'],
        audio_mean=config.audio_mean, audio_std=config.audio_std, datasets=parent['datasets'],
        protected=protected, control=str(CONTROL), control_loro=str(LORO),
        mapping='Normalized log-mel column (128 bins) -> Linear(128,128) -> 128 foreground logits at each 5 ms; no encoder weights, hidden activation, learned positions or temporal mixing.',
        matching='Same 2000 s training/800 s validation source splits, hard-mask BCE, optimizer, five epochs and best XC validation selection as seed-0 SongMAE control. Seed is shared; exact initialization or minibatch ordering is not asserted across different architectures.',
        inference='Same full-recording frontend, five-second windows, 2.5-second stride, maximum overlap and float16 autocast as control. Gaussian probability sigma (2 mel,3 frames); no other cleanup.',
        reporting='All 77 Powdermill segments; leave one original recording out for threshold calibration. Also retain Recording_1-only legacy report. No external benchmarks or website changes.',
        caveat='Single-seed exploratory baseline. Different input dimension, parameter count and receptive field; not an isolated test of pretraining versus random encoder features.')
    path = OUT / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('frozen raw-spectrogram experiment changed')
        return plan
    for key, bins in [('new2k', 400000), ('validation', 160000)]:
        expected = plan['datasets'][key]
        data = PixelWindows(read_rows(expected['path'], 0), ROOT / 'data/xcl/shards', config)
        inputs, masks = hashlib.sha256(), hashlib.sha256()
        if sum(w[2] for w in data.windows) != bins:
            raise ValueError('incorrect training or validation duration')
        for spec, target, valid in data:
            if not torch.isfinite(spec).all() or not torch.all((target == 0) | (target == 1)):
                raise ValueError('invalid inputs or nonbinary supervision')
            inputs.update(spec.numpy().tobytes())
            masks.update(target[:, :valid].numpy().tobytes())
        if inputs.hexdigest() != expected['preflight_input_sha256'] or masks.hexdigest() != expected['preflight_mask_sha256']:
            raise ValueError('normalized spectrograms or targets differ from SongMAE training')
    write_json(path, plan)
    return plan


def train(plan):
    audit = OUT / 'checkpoint.json'
    if CHECKPOINT.exists():
        if not audit.exists() or digest(CHECKPOINT) != json.loads(audit.read_text())['sha256']:
            raise ValueError('completed checkpoint provenance missing or changed')
        return
    progress('training', epoch_budget=5)
    pending = CHECKPOINT.with_suffix('.training.pt')
    if not pending.exists():
        trainer.load_backbone = lambda model_id, device, revision=None: frontend(model_id, revision).to(device).eval()
        trainer.DenseHead = SpectrogramLinear
        sys.argv = ['scripts/train.py', '--annotations', plan['datasets']['new2k']['path'],
            '--validation-annotations', plan['datasets']['validation']['path'],
            '--backbone', plan['frontend_reference'], '--backbone-revision', plan['frontend_revision'],
            '--hidden', '0', '--head-layers', '0', '--seed', '0', '--epochs', '5',
            '--learning-rate', '.001', '--weight-decay', '.0001', '--batch-size', '16',
            '--eval-batch-size', '4', '--dropout', '0', '--tv-weight', '0',
            '--no-target-smoothing', '--log-every', '5', '--out', str(pending)]
        trainer.main()
    saved = torch.load(pending, map_location='cpu', weights_only=True)
    if saved['metrics']['train_timebins'] != 400000 or saved['metrics']['validation_timebins'] != 160000:
        raise ValueError('unmatched training budget')
    SpectrogramLinear().load_state_dict(saved['head'])
    if sum(p.numel() for p in saved['head'].values()) != 16512:
        raise ValueError('unexpected raw-spectrogram model size')
    saved.update(backbone_id=None, backbone_revision=None, detector_architecture=ARCHITECTURE,
        frontend_reference=plan['frontend_reference'], frontend_revision=plan['frontend_revision'],
        backbone_weights_used=False, manifest_sha256=digest(OUT / 'manifest.json'))
    torch.save(saved, CHECKPOINT)
    write_json(audit, dict(path=str(CHECKPOINT), sha256=digest(CHECKPOINT),
        metrics=saved['metrics'], history=saved['training_history'], parameters=16512))


def evaluate(plan):
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    control = json.loads(CONTROL.read_text())
    anchor = json.loads(Path(control['protocol']['anchor']).read_text())['protocol']
    for name, key in [('wav_Files.zip', 'audio_sha256'), ('annotation_Files.zip', 'references_sha256')]:
        if digest(RAW / 'powdermill' / name) != anchor[key]:
            raise ValueError('Powdermill audio or references changed')
    device = torch.device('cuda:0')
    front = frontend(plan['frontend_reference'], plan['frontend_revision']).to(device).eval()
    head = SpectrogramLinear().to(device).eval()
    head.load_state_dict(saved['head'])
    inventory = {r['name']:r for r in recordings(RAW, 'powdermill')}
    scored, completed = {}, 0
    for part, rows in control['per_recording'].items():
        scored[part] = []
        for old in rows:
            name = old['name']
            path = OUT / 'segments' / f'{name}.json'
            source = dict(checkpoint_sha256=digest(CHECKPOINT), manifest_sha256=digest(OUT / 'manifest.json'),
                reference_sha256=hashlib.sha256(inventory[name]['events'].tobytes()).hexdigest())
            if path.exists():
                item = json.loads(path.read_text())
                if item['source'] != source:
                    raise ValueError('segment cache provenance changed')
                value = item['score']
            else:
                wave, rate = read_audio(inventory[name])
                duration = len(wave) / rate
                audio, _ = model_audio(wave, rate, 'songmae')
                probability = songmae_probability(front, head, audio, device)
                truth, kept = truth_and_coverage(inventory[name], round(duration * 200), [[0, duration]])
                if not kept.all():
                    raise ValueError('incomplete Powdermill coverage')
                value = dict(name=name, group=inventory[name]['group'], seconds=duration,
                    area={'full': area(smooth(probability)[:, :len(kept)], truth)})
                write_json(path, dict(source=source, score=value,
                    raw_probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest()))
            if value['seconds'] != old['seconds'] or value['area']['full']['counts'][0] != old['area']['full']['counts'][0]:
                raise ValueError('raw-probe and control reference/coverage mismatch')
            scored[part].append(value)
            completed += 1
            progress('evaluating', completed=completed, total=77, segment=name)
            print(f'Powdermill {completed}/77: {name}', flush=True)
    thresholds = calibrate(scored['calibration'])
    report = dict(protocol=plan, per_recording=scored,
        summary=summarize(scored['evaluation'], thresholds)['full'],
        calibration_summary=summarize(scored['calibration'], thresholds)['full'])
    write_json(OUT / 'comparison.json', report)
    spec = dict(id='raw_spectrogram_s0', family='raw_spectrogram', label='Raw spectrogram + linear',
        seed=0, threshold_floor_index=0, path=OUT / 'comparison.json')
    loro = cross_calibrate(spec, report, checked_rows(report))
    write_json(OUT / 'loro.json', loro)
    previous = json.loads(LORO.read_text())
    write_json(OUT / 'summary.json', dict(raw_spectrogram=loro['summary'], songmae_linear_seed0=previous['summary'],
        delta_raw_minus_songmae={k:loro['summary'][k]-previous['summary'][k] for k in ['ap', 'iou']},
        checkpoint=json.loads((OUT / 'checkpoint.json').read_text()), caveat=plan['caveat']))
    text = ['# Linear detection directly on spectrograms', '', plan['mapping'], plan['matching'], '',
        '| Input features (seed 0) | Pixel AP | 2D IoU |', '|---|---:|---:|',
        f"| Raw 128-bin spectrogram | {loro['summary']['ap']:.3f} | {loro['summary']['iou']:.3f} |",
        f"| Frozen SongMAE-Large latents | {previous['summary']['ap']:.3f} | {previous['summary']['iou']:.3f} |", '',
        'All 77 five-minute segments (6 h 25 min), leave-one-original-recording-out threshold calibration. Same probability smoothing; segment-macro metrics. No external dataset evaluation.',
        f"Selected epoch: {saved['metrics']['selected_epoch']} of 5, by XC validation BCE. Raw probe: 16,512 parameters; SongMAE probe: 24,608 plus frozen backbone.",
        plan['caveat'], 'Existing website, figures, benchmark results and annotation services were not changed.']
    (OUT / 'README.md').write_text('\n'.join(text) + '\n')
    print('\n'.join(text), flush=True)


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = prepare()
        if (OUT / 'status.json').exists() and json.loads((OUT / 'status.json').read_text())['state'] == 'complete':
            print('Already complete:', OUT)
            return
        if torch.cuda.mem_get_info(0)[0] < 2 * 1024**3:
            raise ValueError('GPU memory unavailable; preserve other work')
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        train(plan)
        evaluate(plan)
        if any(digest(p) != sha for p, sha in plan['protected'].items()):
            raise ValueError('frozen experiment input changed')
        progress('complete', completed=77, total=77, summary=str(OUT / 'summary.json'))
    except Exception as error:
        progress('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
