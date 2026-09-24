#!/usr/bin/env python3
"""Unfreeze only the last encoder block; paired seed-0 training and full Powdermill scoring."""
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluate_current_backbones as scorer
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.data import PixelWindows, read_rows
from birdsong_detect_distill.last_block_linear import ARCHITECTURE, BLOCK, last_block, load_heads
from birdsong_detect_distill.model import load_backbone, loss
from birdsong_detect_distill.pointwise_bare_linear import BareLinearHead
from prepare_detector_study import verify
from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate
from train import RecordedSampler, evaluate as validate

ROOT = Path(__file__).resolve().parents[1]
RUN = 'last_block_linear_2000train_800val_2026-09-14'
OUT = ROOT / 'results/qwen_teacher_powdermill' / RUN
ARTIFACTS = Path('/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments') / RUN
PARENT = OUT.parent / 'new_teacher_2000train_800val_2026-09-14/manifest.json'
CONTROL = OUT.parent / 'pointwise_bare_linear_2000train_800val_2026-09-14/runs/bare_linear_d0_new2k_s0/comparison.json'
LORO = OUT.parent / 'leave_one_recording_out_2026-09-14/linear_s0.json'
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
CHECKPOINT = ARTIFACTS / 'last_block_linear_s0.pt'


def progress(state, **values):
    write_json(OUT / 'status.json', dict(state=state, updated_unix=time.time(), **values))


def state_hash(state):
    value = hashlib.sha256()
    for tensor in state.values():
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def save_atomic(value, path):
    pending = path.with_suffix('.pending.pt')
    torch.save(value, pending)
    pending.replace(path)


def prepare():
    parent = json.loads(PARENT.read_text())
    verify(parent)
    control = json.loads(CONTROL.read_text())['protocol']['checkpoint']
    paths = [Path(__file__), PARENT, CONTROL, LORO, Path(control['checkpoint']),
        ROOT / 'scripts/evaluate_current_backbones.py', ROOT / 'scripts/summarize_powdermill_loro.py']
    paths += list((ROOT / 'src/birdsong_detect_distill').glob('pointwise_*.py'))
    paths += [ROOT / 'src/birdsong_detect_distill/last_block_linear.py']
    plan = dict(run=RUN, architecture=ARCHITECTURE, unfrozen_module=BLOCK, seed=0, epochs=5,
        backbone=parent['backbone'], revision=parent['revision'], datasets=parent['datasets'],
        training_config=dict(optimizer='AdamW', learning_rate=.001, backbone_learning_rate=.00001,
            weight_decay=.0001, batch_size=16, accumulation=1, validation_batch_size=4,
            target_smoothing=False, tv_weight=0, confidence_targets=False, ignore_uncertain=False),
        backbone_mode='eval throughout, including trainable last block: preserve control dropout behavior',
        initialization='Original pretrained backbone; identical untrained seed-0 bare-linear head and minibatch order. No trained head warm-start.',
        checkpoint_selection='Lowest XC validation hard-mask BCE within five epochs; no Powdermill checkpoint selection.',
        inference='Unchanged: 5-second windows, 2.5-second stride, maximum overlap; Gaussian probabilities sigma=(2 mel,3 frames); single calibrated threshold.',
        reporting='All 77 Powdermill segments, leave-one-original-recording-out threshold calibration; also retain Recording_1 legacy scores.',
        caveat='Single-seed exploratory development experiment, not independent testing or a learning-rate sweep.',
        protected_files={**parent['protected_files'], **{str(p):digest(p) for p in paths}},
        code_sha256=parent['code_sha256'], control=str(CONTROL), control_loro=str(LORO))
    destination = OUT / 'manifest.json'
    if destination.exists() and json.loads(destination.read_text()) != plan:
        raise ValueError('frozen experiment changed')
    write_json(destination, plan)
    return plan


def dataset(plan, key, config):
    expected = plan['datasets'][key]
    rows = read_rows(expected['path'], 0)
    data = PixelWindows(rows, ROOT / 'data/xcl/shards', config)
    if (sorted({r['recording'] for r in rows}) != expected['recording_ids']
            or sum(w[2] for w in data.windows) != expected['seconds'] * 200):
        raise ValueError('data split or duration changed')
    inputs, targets = hashlib.sha256(), hashlib.sha256()
    for spec, target, valid in data:
        if not torch.isfinite(spec).all() or not torch.all((target == 0) | (target == 1)):
            raise ValueError('invalid input or nonbinary target')
        inputs.update(spec.numpy().tobytes())
        targets.update(target[:, :valid].numpy().tobytes())
    if inputs.hexdigest() != expected['preflight_input_sha256'] or targets.hexdigest() != expected['preflight_mask_sha256']:
        raise ValueError('input/target tensors differ from frozen control')
    return data


def train(plan):
    manifest_sha = digest(OUT / 'manifest.json')
    if CHECKPOINT.exists():
        if (torch.load(CHECKPOINT, map_location='cpu', weights_only=True)['manifest_sha256'] != manifest_sha
                or digest(CHECKPOINT) != json.loads((OUT / 'checkpoint.json').read_text())['sha256']):
            raise ValueError('completed checkpoint provenance changed')
        return
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    backbone = load_backbone(plan['backbone'], device, plan['revision'])
    block = last_block(backbone).requires_grad_(True)
    data = dataset(plan, 'new2k', backbone.config)
    head = BareLinearHead(768, 0, 4, 1000, 32, 1, dropout=.1, layers=0).to(device)
    control = json.loads(CONTROL.read_text())['protocol']['checkpoint']
    initial_head = state_hash(head.state_dict())
    if initial_head != control['initial_head_sha256']:
        raise ValueError('initial head differs from seed-0 frozen control')
    frozen = lambda: {k:v for k,v in backbone.state_dict().items() if not k.startswith(BLOCK + '.')}
    audit = dict(initial_head_sha256=initial_head, initial_block_sha256=state_hash(block.state_dict()),
        frozen_backbone_sha256=state_hash(frozen()),
        trainable_backbone_names=[n for n,p in backbone.named_parameters() if p.requires_grad],
        block_parameters=sum(p.numel() for p in block.parameters()), head_parameters=sum(p.numel() for p in head.parameters()))
    if any(not n.startswith(BLOCK + '.') for n in audit['trainable_backbone_names']):
        raise ValueError('parameters outside the final encoder block were unfrozen')
    settings = plan['training_config']
    optimizer = torch.optim.AdamW([{'params':head.parameters(), 'lr':settings['learning_rate']},
        {'params':block.parameters(), 'lr':settings['backbone_learning_rate']}], weight_decay=settings['weight_decay'])
    sampler = RecordedSampler(data)
    loader = DataLoader(data, 16, sampler=sampler, num_workers=2, pin_memory=True)
    val = dataset(plan, 'validation', backbone.config)
    val_loader = DataLoader(val, 4, num_workers=2, pin_memory=True)
    history, best, best_loss = [], None, float('inf')
    resume = ARTIFACTS / 'resume.pt'
    if resume.exists():
        previous = torch.load(resume, map_location='cpu', weights_only=True)
        if previous['manifest_sha256'] != manifest_sha:
            raise ValueError('resume provenance changed')
        block.load_state_dict(previous['last_block'])
        head.load_state_dict(previous['head'])
        optimizer.load_state_dict(previous['optimizer'])
        history, best, best_loss, audit = [previous[k] for k in ['history','best','best_loss','audit']]
        torch.set_rng_state(previous['cpu_rng'])
        torch.cuda.set_rng_state(previous['cuda_rng'], device)
    print(f"Trainable: {audit['block_parameters']:,} last-block + {audit['head_parameters']:,} linear-head parameters", flush=True)
    for epoch in range(len(history) + 1, 6):
        head.train()
        losses = []
        progress('training', epoch=epoch, epochs=5, step=0, steps=len(loader))
        for step, (specs, targets, valid) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            # Deliberately no no_grad: only the final block has trainable parameters.
            tokens = backbone(input_values=specs.to(device), valid_timebins=valid.to(device)).last_hidden_state
            value = loss(head(tokens.float(), valid), targets.to(device), valid, smooth=False, tv_weight=0)
            if not torch.isfinite(value):
                raise ValueError('nonfinite training loss')
            value.backward()
            if epoch == 1 and step == 1:
                gradients = {f'backbone.{n}':p.grad for n,p in backbone.named_parameters() if p.requires_grad}
                gradients.update({f'head.{n}':p.grad for n,p in head.named_parameters()})
                if any(g is None or not torch.isfinite(g).all() for g in gradients.values()):
                    raise ValueError('missing or nonfinite gradient in trainable parameters')
                if any(p.grad is not None for p in backbone.parameters() if not p.requires_grad):
                    raise ValueError('frozen parameters received gradients')
                audit['first_step_gradient_norms'] = {n:float(g.norm()) for n,g in gradients.items()}
            optimizer.step()
            if epoch == 1 and step == 1:
                if state_hash(block.state_dict()) == audit['initial_block_sha256'] or state_hash(head.state_dict()) == initial_head:
                    raise ValueError('last block or head failed to update')
                audit['first_step_both_modules_updated'] = True
                write_json(OUT / 'initialization.json', audit)
            losses.append(float(value.detach()))
            if step % 5 == 0:
                progress('training', epoch=epoch, epochs=5, step=step, steps=len(loader), training_loss=float(np.mean(losses)))
                print(f'epoch {epoch} step {step}/{len(loader)}: train={np.mean(losses):.4f}', flush=True)
        row = dict(epoch=epoch, mean_training_loss=float(np.mean(losses)), sample_order_sha256=sampler.digest.hexdigest(),
            cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
        if any(row[k] != control['training_history'][epoch-1][k] for k in ['sample_order_sha256','cpu_rng_sha256']):
            raise ValueError('training order or RNG differs from seed-0 frozen control')
        torch.cuda.empty_cache()
        row['validation_loss'], _ = validate(backbone, head, val_loader, val, device, 0, target_smoothing=False, collect_rows=False)
        if not np.isfinite(row['validation_loss']):
            raise ValueError('nonfinite validation loss')
        history.append(row)
        if row['validation_loss'] < best_loss:
            best_loss = row['validation_loss']
            best = dict(epoch=epoch, head=cpu_state(head), last_block=cpu_state(block))
        save_atomic(dict(manifest_sha256=manifest_sha, head=cpu_state(head), last_block=cpu_state(block),
            optimizer=optimizer.state_dict(), history=history, best=best, best_loss=best_loss, audit=audit,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)), resume)
        write_json(OUT / 'history.json', history)
        print(f'epoch {epoch}: train={row["mean_training_loss"]:.4f}, XC validation={row["validation_loss"]:.4f}', flush=True)
    if state_hash(frozen()) != audit['frozen_backbone_sha256']:
        raise ValueError('frozen backbone tensors changed during training')
    saved = dict(detector_architecture=ARCHITECTURE, unfrozen_module=BLOCK, manifest_sha256=manifest_sha,
        head=best['head'], last_block=best['last_block'], backbone_id=plan['backbone'], backbone_revision=plan['revision'],
        seed=0, hidden=0, head_layers=0, height=4, width=1000, dropout=.1, layer_norm=False,
        target_smoothing=False, tv_weight=0, training_config=settings, training_history=history,
        initial_head_sha256=initial_head, audit={**audit, 'frozen_tensors_unchanged':True},
        training_recording_ids=plan['datasets']['new2k']['recording_ids'], validation_recording_ids=plan['datasets']['validation']['recording_ids'],
        annotations_sha256=plan['datasets']['new2k']['sha256'], validation_annotations_sha256=plan['datasets']['validation']['sha256'],
        metrics=dict(epochs=5, selected_epoch=best['epoch'], best_val_loss=best_loss, train_timebins=400000,
            validation_timebins=160000, checkpoint_selection='minimum_recording_disjoint_validation_loss'))
    save_atomic(saved, CHECKPOINT)
    write_json(OUT / 'checkpoint.json', dict(path=str(CHECKPOINT), sha256=digest(CHECKPOINT), metrics=saved['metrics'], audit=saved['audit']))


def evaluate(plan):
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    control = json.loads(CONTROL.read_text())
    anchor_path = Path(control['protocol']['anchor'])
    anchor = json.loads(anchor_path.read_text())
    reference = anchor['protocol']
    for name, key in [('wav_Files.zip','audio_sha256'), ('annotation_Files.zip','references_sha256')]:
        if digest(RAW / 'powdermill' / name) != reference[key]:
            raise ValueError('Powdermill inputs changed')
    protocol = dict(manifest=plan, checkpoint={k:v for k,v in saved.items() if k not in ['head','last_block']},
        checkpoint_path=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT), anchor=str(anchor_path),
        anchor_sha256=digest(anchor_path), student_inference=control['protocol']['student_inference'])
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('evaluation provenance changed')
    write_json(path, protocol)
    progress('evaluating', total=77)
    scorer.load_heads = load_heads  # Restores the trained block before the shared inference routine.
    scorer.score_checkpoint(CHECKPOINT, saved, OUT, ARTIFACTS / 'predictions', RAW, reference, anchor, 'samples')
    report = json.loads((OUT / 'comparison.json').read_text())
    rows = checked_rows(report)
    old_rows = {r['name']:r for r in checked_rows(control)}
    if any(r['area']['full']['counts'][0] != old_rows[r['name']]['area']['full']['counts'][0] for r in rows):
        raise ValueError('fine-tuned and frozen model reference masks differ')
    spec = dict(id='last_block_linear_s0', family='last_block_linear', label='Last block unfrozen + linear',
        seed=0, threshold_floor_index=0, path=OUT / 'comparison.json')
    loro = cross_calibrate(spec, report, rows)
    write_json(OUT / 'loro.json', loro)
    previous = json.loads(LORO.read_text())
    summary = dict(last_block_unfrozen=loro['summary'], frozen_seed0=previous['summary'],
        delta={k:loro['summary'][k]-previous['summary'][k] for k in ['ap','iou']},
        selected_epoch=saved['metrics']['selected_epoch'], caveat=plan['caveat'])
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        progress('preparing')
        train(plan)
        evaluate(plan)
        verify(plan)
        progress('complete', completed=77, total=77, summary=str(OUT / 'summary.json'))
    except Exception as error:
        progress('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
