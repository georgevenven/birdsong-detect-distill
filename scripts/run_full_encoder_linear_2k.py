#!/usr/bin/env python3
"""Fine-tune the complete detection encoder on the matched 2,000-second XC split."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluate_current_backbones as scorer
import run_last_block_linear_2k as previous
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.full_encoder_linear import ARCHITECTURE, PARTS, encoder_key, encoder_state, restore_encoder, load_heads
from birdsong_detect_distill.model import load_backbone, loss
from birdsong_detect_distill.pointwise_bare_linear import BareLinearHead
from prepare_detector_study import verify
from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate
from train import RecordedSampler, evaluate as validate

ROOT, CONTROL, LORO, RAW = previous.ROOT, previous.CONTROL, previous.LORO, previous.RAW
RUN = 'full_encoder_linear_2000train_800val_2026-09-14'
OUT, ARTIFACTS = previous.OUT.parent / RUN, previous.ARTIFACTS.parent / RUN
CHECKPOINT = ARTIFACTS / 'full_encoder_linear_s0.pt'
state_hash, cpu_state, save_atomic = previous.state_hash, previous.cpu_state, previous.save_atomic


def progress(state, **values):
    write_json(OUT / 'status.json', dict(state=state, updated_unix=time.time(), **values))


def prepare():
    parent = json.loads((previous.OUT / 'manifest.json').read_text())
    verify(parent)
    if json.loads((previous.OUT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('last-block comparison is not complete')
    paths = [Path(__file__), ROOT / 'src/birdsong_detect_distill/full_encoder_linear.py',
        previous.OUT / 'manifest.json', previous.OUT / 'summary.json', previous.OUT / 'loro.json', previous.CHECKPOINT]
    plan = {**parent, 'run':RUN, 'architecture':ARCHITECTURE, 'unfrozen_module':list(PARTS),
        'backbone_mode':'eval throughout: preserve disabled dropout while training all detection-path parameters',
        'training_config':{**parent['training_config'], 'microbatch_size':2, 'accumulation':8},
        'accumulation':'Same logical batch of 16 and optimizer steps. Backpropagate FP32 microbatches of 2, weighted by valid time bins. No mixed precision or gradient clipping.',
        'excluded':'Only the unused reconstruction decoder, encoder-to-decoder projection and mask tokens remain frozen.',
        'protected_files':{**parent['protected_files'], **{str(p):digest(p) for p in paths}},
        'last_block_control':str(previous.OUT / 'loro.json'),
        'caveat':'Single seed, same 1e-5 encoder LR and 1e-3 head LR as last-block pilot. Gradient accumulation can introduce small numerical differences; no additional learning-rate tuning.'}
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('frozen experiment changed')
    write_json(path, plan)
    return plan


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
    if len(backbone.songmae.encoder.layers) != 12 or backbone.config.enc_hidden_d != 768:
        raise ValueError('expected SongMAE-Large')
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(encoder_key(name))
    data = previous.dataset(plan, 'new2k', backbone.config)
    head = BareLinearHead(768, 0, 4, 1000, 32, 1, dropout=.1, layers=0).to(device)
    control = json.loads(CONTROL.read_text())['protocol']['checkpoint']
    initial_head = state_hash(head.state_dict())
    if initial_head != control['initial_head_sha256']:
        raise ValueError('head initialization differs from frozen control')
    frozen = lambda: {k:v for k,v in backbone.state_dict().items() if not encoder_key(k)}
    weights = lambda: {k:v.detach().cpu().clone() for k,v in encoder_state(backbone).items()}
    trainable = {n:p for n,p in backbone.named_parameters() if p.requires_grad}
    initial_parameters = {n:state_hash({n:p}) for n,p in trainable.items()}
    audit = dict(initial_head_sha256=initial_head, initial_encoder_sha256=state_hash(encoder_state(backbone)),
        unused_pretraining_state_sha256=state_hash(frozen()), trainable_backbone_names=list(trainable),
        encoder_parameters=sum(p.numel() for p in trainable.values()), head_parameters=sum(p.numel() for p in head.parameters()))
    settings = plan['training_config']
    optimizer = torch.optim.AdamW([{'params':head.parameters(), 'lr':settings['learning_rate']},
        {'params':list(trainable.values()), 'lr':settings['backbone_learning_rate']}], weight_decay=settings['weight_decay'])
    sampler = RecordedSampler(data)
    loader = DataLoader(data, 16, sampler=sampler, num_workers=2, pin_memory=True)
    val = previous.dataset(plan, 'validation', backbone.config)
    val_loader = DataLoader(val, 4, num_workers=2, pin_memory=True)
    history, best, best_loss = [], None, float('inf')
    resume = ARTIFACTS / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['manifest_sha256'] != manifest_sha:
            raise ValueError('resume provenance changed')
        restore_encoder(backbone, saved['encoder'])
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, best_loss, audit = [saved[k] for k in ['history','best','best_loss','audit']]
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'], device)
        del saved
    print(f"Trainable: {audit['encoder_parameters']:,} encoder + {audit['head_parameters']:,} head; microbatch 2, effective batch 16", flush=True)
    for epoch in range(len(history) + 1, 6):
        head.train()
        losses = []
        progress('training', epoch=epoch, epochs=5, step=0, steps=len(loader))
        for step, (specs, targets, valid) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            total_bins, batch_loss = int(valid.sum()), 0.0
            for start in range(0, len(specs), settings['microbatch_size']):
                section = slice(start, start + settings['microbatch_size'])
                weight = int(valid[section].sum()) / total_bins
                tokens = backbone(input_values=specs[section].to(device), valid_timebins=valid[section].to(device)).last_hidden_state
                value = loss(head(tokens.float(), valid[section]), targets[section].to(device), valid[section], smooth=False, tv_weight=0)
                if not torch.isfinite(value):
                    raise ValueError('nonfinite training loss')
                (value * weight).backward()
                batch_loss += float(value.detach()) * weight
                del tokens, value
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in [*trainable.values(), *head.parameters()]):
                raise ValueError('missing/nonfinite gradient in the full encoder or head')
            if epoch == 1 and step == 1:
                audit['first_step_gradient_norms'] = {n:float(p.grad.norm()) for n,p in trainable.items()}
                if any(p.grad is not None for p in backbone.parameters() if not p.requires_grad):
                    raise ValueError('unused pretraining weights received gradients')
            optimizer.step()
            if epoch == 1 and step == 1:
                unchanged = [n for n,p in trainable.items() if state_hash({n:p}) == initial_parameters[n]]
                if unchanged or state_hash(head.state_dict()) == initial_head:
                    raise ValueError(f'parameters did not update: {unchanged}')
                audit.update(all_encoder_parameters_updated=True, head_updated=True,
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
                write_json(OUT / 'initialization.json', audit)
                print('Verified gradients and weight updates throughout the encoder and head.', flush=True)
            losses.append(batch_loss)
            if step % 5 == 0:
                progress('training', epoch=epoch, epochs=5, step=step, steps=len(loader), training_loss=float(np.mean(losses)))
                print(f'epoch {epoch} step {step}/{len(loader)}: train={np.mean(losses):.4f}', flush=True)
        row = dict(epoch=epoch, mean_training_loss=float(np.mean(losses)), sample_order_sha256=sampler.digest.hexdigest(),
            cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
        if any(row[k] != control['training_history'][epoch-1][k] for k in ['sample_order_sha256','cpu_rng_sha256']):
            raise ValueError('sample order or CPU RNG differs from frozen control')
        torch.cuda.empty_cache()
        row['validation_loss'], _ = validate(backbone, head, val_loader, val, device, 0, target_smoothing=False, collect_rows=False)
        if not np.isfinite(row['validation_loss']):
            raise ValueError('nonfinite validation loss')
        history.append(row)
        if row['validation_loss'] < best_loss:
            best_loss = row['validation_loss']
            best = dict(epoch=epoch, head=cpu_state(head), encoder=weights())
        save_atomic(dict(manifest_sha256=manifest_sha, head=cpu_state(head), encoder=weights(),
            optimizer=optimizer.state_dict(), history=history, best=best, best_loss=best_loss, audit=audit,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)), resume)
        write_json(OUT / 'history.json', history)
        print(f'epoch {epoch}: train={row["mean_training_loss"]:.4f}, XC validation={row["validation_loss"]:.4f}', flush=True)
    if state_hash(frozen()) != audit['unused_pretraining_state_sha256']:
        raise ValueError('unused pretraining weights changed')
    saved = dict(detector_architecture=ARCHITECTURE, manifest_sha256=manifest_sha,
        head=best['head'], encoder=best['encoder'], backbone_id=plan['backbone'], backbone_revision=plan['revision'],
        seed=0, hidden=0, head_layers=0, height=4, width=1000, dropout=.1, layer_norm=False,
        target_smoothing=False, tv_weight=0, training_config=settings, training_history=history,
        initial_head_sha256=initial_head, audit={**audit, 'unused_pretraining_weights_unchanged':True},
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
    protocol = dict(manifest=plan, checkpoint={k:v for k,v in saved.items() if k not in ['head','encoder']},
        checkpoint_path=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT), anchor=str(anchor_path),
        anchor_sha256=digest(anchor_path), student_inference=control['protocol']['student_inference'])
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('evaluation provenance changed')
    write_json(path, protocol)
    progress('evaluating', total=77)
    scorer.load_heads = load_heads
    scorer.score_checkpoint(CHECKPOINT, saved, OUT, ARTIFACTS / 'predictions', RAW, reference, anchor, 'samples')
    report = json.loads((OUT / 'comparison.json').read_text())
    rows = checked_rows(report)
    old_rows = {r['name']:r for r in checked_rows(control)}
    if any(r['area']['full']['counts'][0] != old_rows[r['name']]['area']['full']['counts'][0] for r in rows):
        raise ValueError('reference masks differ from frozen control')
    spec = dict(id='full_encoder_linear_s0', family='full_encoder_linear', label='Full encoder unfrozen + linear',
        seed=0, threshold_floor_index=0, path=OUT / 'comparison.json')
    loro = cross_calibrate(spec, report, rows)
    write_json(OUT / 'loro.json', loro)
    controls = dict(frozen_seed0=json.loads(LORO.read_text())['summary'],
        last_block_seed0=json.loads((previous.OUT / 'loro.json').read_text())['summary'])
    summary = dict(full_encoder=loro['summary'], **controls,
        deltas={name:{k:loro['summary'][k]-value[k] for k in ['ap','iou']} for name,value in controls.items()},
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
        if shutil.disk_usage(ARTIFACTS).free < 8*1024**3:
            raise ValueError('insufficient checkpoint disk space')
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
