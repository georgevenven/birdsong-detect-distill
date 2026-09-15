#!/usr/bin/env python3
"""Reuse the pinned full-fine-tuning loop with only transformer blocks trainable."""
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

import run_full_encoder_linear_2k as engine
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.transformer_blocks_linear import ARCHITECTURE, encoder_key, encoder_state, restore_encoder, load_heads
from prepare_detector_study import verify
from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate

ROOT, PARENT = engine.ROOT, engine.OUT
RUN = 'transformer_blocks_linear_2000train_800val_2026-09-14'
OUT, ARTIFACTS = PARENT.parent / RUN, engine.ARTIFACTS.parent / RUN
CHECKPOINT = ARTIFACTS / 'transformer_blocks_linear_s0.pt'


def prepare():
    parent = json.loads((PARENT / 'manifest.json').read_text())
    verify(parent)
    if json.loads((PARENT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('full-fine-tuning control must be complete')
    full_checkpoint = json.loads((PARENT / 'checkpoint.json').read_text())
    paths = [Path(__file__), ROOT / 'src/birdsong_detect_distill/transformer_blocks_linear.py',
        PARENT / 'manifest.json', PARENT / 'comparison.json', PARENT / 'loro.json',
        PARENT / 'summary.json', Path(full_checkpoint['path'])]
    plan = {**parent, 'run':RUN, 'architecture':ARCHITECTURE, 'unfrozen_module':['songmae.encoder'],
        'backbone_mode':'eval throughout: disabled dropout, gradients only in the twelve transformer blocks',
        'excluded':'Original pretrained patch projection, both patch convolutions, frequency/time embeddings, plus unused decoder and mask tokens remain frozen.',
        'initialization':'Original pretrained SongMAE, not the fully fine-tuned checkpoint. Same untrained seed-0 linear head and window order as controls.',
        'protected_files':{**parent['protected_files'], **{str(p):digest(p) for p in paths}},
        'full_encoder_control':str(PARENT / 'loro.json'),
        'caveat':'Single-seed development pilot. Same exact training loop, microbatches, learning rates and inference as full fine-tuning; only trainable parameter scope changes.'}
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('frozen experiment changed')
    write_json(path, plan)
    return plan


def train(plan):
    # Process-local injection avoids editing either control's hash-pinned implementation.
    engine.ARCHITECTURE = ARCHITECTURE
    engine.encoder_key, engine.encoder_state = encoder_key, encoder_state
    engine.restore_encoder = restore_encoder
    engine.train(plan)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    audit = saved['audit']
    expected = json.loads((PARENT / 'protocol.json').read_text())['checkpoint']
    if (saved['initial_head_sha256'] != expected['initial_head_sha256']
            or saved['training_config'] != expected['training_config']
            or len(saved['training_history']) != 5):
        raise ValueError('initialization or training settings differ from full-fine-tuning control')
    for a,b in zip(saved['training_history'], expected['training_history']):
        if any(a[k] != b[k] for k in ['epoch','sample_order_sha256','cpu_rng_sha256']):
            raise ValueError('training order or RNG differs from full-fine-tuning control')
    if (audit['encoder_parameters'] != 85054464 or audit['head_parameters'] != 24608
            or any(not encoder_key(n) for n in audit['trainable_backbone_names'])
            or any(not encoder_key(n) for n in saved['encoder'])
            or not audit['unused_pretraining_weights_unchanged']):
        raise ValueError('incorrect trainable/frozen scope')
    # The reused engine hashes ALL non-trainable tensors under its historical key.
    # Here this explicitly includes the CNN, patch projection and positional embeddings.
    write_json(OUT / 'scope_audit.json', dict(
        frozen_frontend_and_decoder_bitwise_unchanged=True,
        frozen_scope=plan['excluded'], frozen_state_sha256=audit['unused_pretraining_state_sha256'],
        first_step_all_transformer_parameters_updated=audit['all_encoder_parameters_updated'],
        matching_full_control_initialization_settings_and_order=True,
        checkpoint_sha256=digest(CHECKPOINT), encoder_parameters=85054464, head_parameters=24608))


def evaluate(plan):
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    control = json.loads(engine.CONTROL.read_text())
    anchor_path = Path(control['protocol']['anchor'])
    anchor = json.loads(anchor_path.read_text())
    reference = anchor['protocol']
    for name,key in [('wav_Files.zip','audio_sha256'), ('annotation_Files.zip','references_sha256')]:
        if digest(engine.RAW / 'powdermill' / name) != reference[key]:
            raise ValueError('Powdermill inputs changed')
    protocol = dict(manifest=plan, checkpoint={k:v for k,v in saved.items() if k not in ['head','encoder']},
        checkpoint_path=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT), anchor=str(anchor_path),
        anchor_sha256=digest(anchor_path), student_inference=control['protocol']['student_inference'],
        scope_audit=json.loads((OUT / 'scope_audit.json').read_text()))
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('evaluation provenance changed')
    write_json(path, protocol)
    engine.progress('evaluating', total=77)
    engine.scorer.load_heads = load_heads
    engine.scorer.score_checkpoint(CHECKPOINT, saved, OUT, ARTIFACTS / 'predictions',
        engine.RAW, reference, anchor, 'samples')
    report = json.loads((OUT / 'comparison.json').read_text())
    rows = checked_rows(report)
    old_rows = {r['name']:r for r in checked_rows(control)}
    if any(r['area']['full']['counts'][0] != old_rows[r['name']]['area']['full']['counts'][0] for r in rows):
        raise ValueError('reference masks differ from frozen control')
    spec = dict(id='transformer_blocks_linear_s0', family='transformer_blocks_linear',
        label='Transformer blocks unfrozen; CNN and positions frozen + linear',
        seed=0, threshold_floor_index=0, path=OUT / 'comparison.json')
    loro = cross_calibrate(spec, report, rows)
    write_json(OUT / 'loro.json', loro)
    controls = dict(frozen_seed0=json.loads(engine.LORO.read_text())['summary'],
        last_block_seed0=json.loads((engine.previous.OUT / 'loro.json').read_text())['summary'],
        full_encoder_seed0=json.loads((PARENT / 'loro.json').read_text())['summary'])
    summary = dict(transformer_blocks=loro['summary'], **controls,
        deltas={name:{k:loro['summary'][k]-value[k] for k in ['ap','iou']} for name,value in controls.items()},
        selected_epoch=saved['metrics']['selected_epoch'], caveat=plan['caveat'])
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    engine.OUT, engine.ARTIFACTS, engine.CHECKPOINT = OUT, ARTIFACTS, CHECKPOINT
    try:
        plan = prepare()
        if (OUT / 'status.json').exists() and json.loads((OUT / 'status.json').read_text())['state'] == 'complete':
            return
        for suffix in ['twins','server']:
            state = subprocess.check_output(['systemctl','--user','show',
                f'birdsong-qwen-xc50k-{suffix}-20260914.service','-p','ActiveState','--value'], text=True).strip()
            if state != 'inactive':
                raise ValueError('Twins Qwen is not paused; preserve existing work')
        if torch.cuda.mem_get_info(0)[0] < 20*1024**3:
            raise ValueError('GPU occupied; preserve existing work')
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(ARTIFACTS).free < 8*1024**3:
            raise ValueError('insufficient checkpoint disk space')
        engine.progress('preparing')
        train(plan)
        evaluate(plan)
        verify(plan)
        engine.progress('complete', completed=77, total=77, summary=str(OUT / 'summary.json'))
    except Exception as error:
        engine.progress('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
