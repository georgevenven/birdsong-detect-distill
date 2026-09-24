#!/usr/bin/env python3
"""Scaling study seeds 1-2: new default recipe at 250/1k/2.5k/5k/10k/25k s, micro/base/large."""
import argparse
import fcntl
import hashlib
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from run_v100_lr_sweep import frozen, read, sha, write

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/v100_lr_bundle_20260916')
REPO = Path('/home/george-vengrovski/Documents/birdsong-detect-distill')
LABELS = REPO / 'data/annotations/xcl/paper_25000train_3seeds_10epochs_2026-09-15'
SHARDS = REPO / 'data/xcl/shards'
OUT = ROOT / 'results'
BUDGETS = [250, 1000, 2500, 5000, 10000, 25000]
SIZES = ['micro', 'base', 'large']
SEEDS = [1, 2]
JOBS = {f'{size}_{budget}_s{seed}': dict(size=size, budget=budget, seed=seed)
    for seed in SEEDS for size in SIZES for budget in BUDGETS}


def prepare():
    import numpy as np
    import torch
    from birdsong_detect_distill.data import PixelWindows, read_rows
    from birdsong_detect_distill.full_encoder_linear import encoder_state
    from birdsong_detect_distill.model import load_backbone
    from birdsong_detect_distill.pointwise_bare_linear import BareLinearHead
    from run_last_block_linear_2k import state_hash
    if (OUT / 'manifest.json').exists():
        plan = read(OUT / 'manifest.json')
        for path, expected in plan['protected_files'].items():
            if sha(path) != expected:
                raise ValueError('Frozen input changed: ' + str(path))
        return plan
    paper = read(REPO / 'results/paper_25000train_3seeds_10epochs_2026-09-15/manifest.json')
    master = read(SOURCE / 'source_manifest.json')
    models = master['models']
    protected = {}
    torch.manual_seed(0)
    probe = load_backbone(models['large']['backbone'], torch.device('cpu'), models['large']['revision'])
    config = probe.config
    del probe
    full_rows = read_rows(LABELS / '25000.jsonl', 0)
    full_order = [r['recording'] for r in full_rows]
    if len(full_rows) != 5000:
        raise ValueError('Wrong 25k pool size')
    import json as _json
    raw_order = [ _json.loads(l)['recording'] for l in open(LABELS / '25000.jsonl') if _json.loads(l).get('status') == 'ok' ]
    budgets = {}
    for budget in BUDGETS:
        windows = budget // 5
        if budget in (100, 1000, 5000, 10000, 25000):
            expected = paper['datasets'][str(budget)]
            labels = LABELS / f'{budget}.jsonl'
            if sha(labels) != expected['sha256']:
                raise ValueError('Label selection changed')
            protected[str(labels)] = expected['sha256']
            pinned = [ _json.loads(l)['recording'] for l in open(labels) if _json.loads(l).get('status') == 'ok' ]
            if pinned != raw_order[:windows]:
                raise ValueError(f'Budget {budget} not a nested prefix')
            if len(pinned) * 5 != expected['seconds']:
                raise ValueError('Wrong duration')
        budgets[str(budget)] = dict(windows=windows, recordings=full_order[:windows])
    expected = paper['datasets']['validation']
    labels = LABELS / 'validation.jsonl'
    if sha(labels) != expected['sha256']:
        raise ValueError('Label selection changed')
    protected[str(labels)] = expected['sha256']
    val_rows = read_rows(labels, 0)
    if sorted({r['recording'] for r in val_rows}) != expected['recording_ids']:
        raise ValueError('Wrong validation recordings')
    pool = PixelWindows(full_rows, SHARDS, config)
    a, b = hashlib.sha256(), hashlib.sha256()
    for spec, target, valid in pool:
        a.update(spec.numpy().tobytes())
        b.update(target.numpy().astype(np.float32).tobytes())
    if (a.hexdigest(), b.hexdigest()) != (paper['datasets']['25000']['preflight_input_sha256'], paper['datasets']['25000']['preflight_mask_sha256']):
        raise ValueError('Train tensor mismatch')
    pool = PixelWindows(val_rows, SHARDS, config)
    a, b = hashlib.sha256(), hashlib.sha256()
    for spec, target, valid in pool:
        a.update(spec.numpy().tobytes())
        b.update(target.numpy().astype(np.float32).tobytes())
    if (a.hexdigest(), b.hexdigest()) != (expected['preflight_input_sha256'], expected['preflight_mask_sha256']):
        raise ValueError('Validation tensor mismatch')
    if set(full_order) & set(expected['recording_ids']):
        raise ValueError('Recording leakage')
    data = dict(budgets=budgets, validation=dict(expected, path=str(labels)))
    initializations = {}
    for size in ['micro', 'base', 'large']:
        model = models[size]
        backbone = load_backbone(model['backbone'], torch.device('cpu'), model['revision'])
        enc_sha = state_hash(encoder_state(backbone))
        depth = len(backbone.songmae.encoder.layers)
        hidden = backbone.config.enc_hidden_d
        del backbone
        for seed in SEEDS:
            torch.manual_seed(seed)
            head = BareLinearHead(hidden, 0, 4, 1000, 32, 1, dropout=.1, layers=0)
            initializations[f'{size}_s{seed}'] = dict(
                initial_head_sha256=state_hash(head.state_dict()),
                initial_encoder_sha256=enc_sha, encoder_depth=depth)
            del head
    if initializations['large_s1']['initial_encoder_sha256'] != 'd6c687d829ca99c8b2e37b2e2710e3e1827ede991f0e56ff3d3416ec9d2cadf1':
        raise ValueError('Pretrained Large encoder differs')
    if not initializations['base_s1']['initial_encoder_sha256'].startswith('59284e98425daf1a'):
        raise ValueError('Pretrained Base encoder differs')
    plan = dict(run='scaling_default_seeds12_2026-09-18', seeds=SEEDS, epochs=5,
        models=models, datasets=data, jobs=JOBS, initializations=initializations,
        training_config=dict(optimizer='AdamW', learning_rate=1e-3, backbone_learning_rate=1e-5,
            weight_decay=1e-4, batch_size=16, microbatch_size=2, target_smoothing=False, tv_weight=0),
        recipe='New default: constant encoder LR 1e-5 (single group, no layerwise), constant head LR 1e-3, no schedule. bf16 autocast forward/backward; fp32 master weights, optimizer, validation. 5 epochs.',
        checkpoint_selection='Minimum recording-disjoint XC validation BCE; every epoch also saved.',
        inference='Per-epoch full Powdermill (77 segments), native frontend, sigma=(2,3), pixel AP + LORO IoU.',
        protected_files=protected,
        code_sha256={str(p.relative_to(ROOT)): sha(p) for folder in ['scripts', 'src'] for p in (ROOT / folder).rglob('*.py')})
    frozen(OUT / 'manifest.json', plan)
    return plan


def set_rates(optimizer, plan, epoch, step, steps):
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr']
    return dict(head=optimizer.param_groups[0]['lr'],
        encoder_min=optimizer.param_groups[1]['lr'], encoder_max=optimizer.param_groups[1]['lr'])


def epoch_save(plan, size, model, audit, history, head_state, enc_state, epoch, val_loss, path):
    from birdsong_detect_distill.full_encoder_linear import ARCHITECTURE
    from run_last_block_linear_2k import save_atomic
    saved = dict(detector_architecture=ARCHITECTURE, manifest_sha256=sha(plan['out_manifest']),
        head=head_state, encoder=enc_state, backbone_id=model['backbone'], backbone_revision=model['revision'],
        seed=plan['seed'], hidden=0, head_layers=0, height=4, width=1000, dropout=.1, layer_norm=False,
        target_smoothing=False, tv_weight=0, training_config=plan['training_config'],
        training_recipe=dict(scheduled=False, layerwise=False, head_only_epochs=0), training_history=list(history),
        initial_head_sha256=audit['initial_head_sha256'], audit=dict(audit),
        training_recording_ids=plan['datasets']['budgets'][str(plan['recipe']['budget'])]['recordings'],
        validation_recording_ids=plan['datasets']['validation']['recording_ids'],
        annotations_sha256='25000-pool-prefix',
        validation_annotations_sha256=plan['datasets']['validation']['sha256'],
        metrics=dict(epochs=plan['epochs'], selected_epoch=epoch, best_val_loss=val_loss, head_only_epochs=0,
            full_finetuning_epochs=plan['epochs'], train_timebins=plan['recipe']['budget'] * 200,
            validation_timebins=2500 * 200,
            checkpoint_selection='per_epoch_trajectory'))
    save_atomic(saved, path)


def train_job(plan, out, artifacts):
    import numpy as np
    import torch
    from birdsong_detect_distill.full_encoder_linear import ARCHITECTURE, encoder_key, encoder_state as backbone_encoder_state, restore_encoder
    from birdsong_detect_distill.model import load_backbone, loss as detection_loss
    from birdsong_detect_distill.pointwise_bare_linear import BareLinearHead
    from run_last_block_linear_2k import cpu_state as cpu, save_atomic as save, state_hash
    from train import RecordedSampler, evaluate as validation_loss
    from torch.utils.data import DataLoader
    size = plan['recipe']['size']
    model = plan['models'][size]
    manifest_sha = sha(out / 'manifest.json')
    plan = dict(plan, out_manifest=out / 'manifest.json')
    torch.manual_seed(plan['seed'])
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    backbone = load_backbone(model['backbone'], device, model['revision'])
    depth = len(backbone.songmae.encoder.layers)
    if depth != plan['initialization']['encoder_depth']:
        raise ValueError('Encoder depth changed')
    head = BareLinearHead(backbone.config.enc_hidden_d, 0, 4, 1000, 32, 1, dropout=.1, layers=0).to(device)
    encoder = {n: p for n, p in backbone.named_parameters() if encoder_key(n)}
    weights = lambda: {n: v.detach().cpu().clone() for n, v in backbone_encoder_state(backbone).items()}
    unused = lambda: {n: v for n, v in backbone.state_dict().items() if not encoder_key(n)}
    hash_state = state_hash
    audit = dict(initial_head_sha256=hash_state(head.state_dict()),
        initial_encoder_sha256=hash_state(backbone_encoder_state(backbone)),
        unused_pretraining_state_sha256=hash_state(unused()),
        encoder_parameters=sum(p.numel() for p in encoder.values()),
        head_parameters=sum(p.numel() for p in head.parameters()),
        trainable_backbone_names=list(encoder))
    for key in ['initial_head_sha256', 'initial_encoder_sha256']:
        if audit[key] != plan['initialization'][key]:
            raise ValueError('Initial weights differ')
    groups = [dict(params=list(head.parameters()), role='head',
        peak_lr=plan['training_config']['learning_rate'], names=list(head.state_dict())),
        dict(params=list(encoder.values()), role='encoder',
            peak_lr=plan['training_config']['backbone_learning_rate'], names=list(encoder))]
    frozen(out / 'optimizer_groups.json', [{k: v for k, v in g.items() if k != 'params'} for g in groups])
    optimizer = torch.optim.AdamW(groups, weight_decay=plan['training_config']['weight_decay'])

    from birdsong_detect_distill.data import PixelWindows, read_rows
    budget = plan['recipe']['budget']
    train_rows = read_rows(LABELS / '25000.jsonl', plan['seed'])[:budget // 5]
    if [r['recording'] for r in train_rows] != plan['datasets']['budgets'][str(budget)]['recordings']:
        raise ValueError('Budget slice changed')
    train_data = PixelWindows(train_rows, SHARDS, backbone.config)
    val_data = PixelWindows(read_rows(plan['datasets']['validation']['path'], plan['seed']), SHARDS, backbone.config)
    data, val = train_data, val_data
    torch.manual_seed(plan['seed'])
    sampler = RecordedSampler(data)
    loader = DataLoader(data, 16, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, 4, num_workers=0, pin_memory=True)
    resume = artifacts / 'resume.pt'
    history, best, best_loss = [], None, float('inf')
    start = 1
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if (saved['manifest_sha256'] != manifest_sha or saved['size'] != size
                or saved.get('budget') != plan['recipe']['budget']):
            raise ValueError('Resume provenance changed')
        restore_encoder(backbone, saved['encoder'])
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, best_loss, audit = [saved[k] for k in ['history', 'best', 'best_loss', 'audit']]
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'], device)
        del saved
        start = len(history) + 1
    write(out / 'initialization.json', audit)
    for epoch in range(start, plan['epochs'] + 1):
        for name, p in backbone.named_parameters():
            p.requires_grad_(encoder_key(name))
        backbone.eval()
        head.train()
        losses = []
        for step, (specs, targets, valid) in enumerate(loader, 1):
            rates = set_rates(optimizer, plan, epoch, step, len(loader))
            if step == 1:
                first_rates = rates
            optimizer.zero_grad(set_to_none=True)
            total = int(valid.sum())
            batch_loss = 0.
            for s in range(0, len(specs), plan['training_config']['microbatch_size']):
                section = slice(s, s + plan['training_config']['microbatch_size'])
                weight = int(valid[section].sum()) / total
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    tokens = backbone(input_values=specs[section].to(device), valid_timebins=valid[section].to(device)).last_hidden_state
                    logits = head(tokens, valid[section])
                value = detection_loss(logits.float(), targets[section].to(device), valid[section], smooth=False, tv_weight=0)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite loss')
                (value * weight).backward()
                batch_loss += float(value.detach()) * weight
                del tokens, value
            optimizer.step()
            losses.append(batch_loss)
            if step % 5 == 0 or step == len(loader):
                print(f"{plan['job']} epoch {epoch} step {step}/{len(loader)} train={np.mean(losses):.4f} lr={rates}", flush=True)
        if hash_state(unused()) != audit['unused_pretraining_state_sha256']:
            raise ValueError('Unused pretraining components changed')
        torch.cuda.empty_cache()
        value, _ = validation_loss(backbone, head, val_loader, val, device, 0, target_smoothing=False, collect_rows=False)
        history.append(dict(epoch=epoch, phase='full_finetuning', mean_training_loss=float(np.mean(losses)),
            validation_loss=value, sample_order_sha256=sampler.digest.hexdigest(),
            cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            learning_rates_first=first_rates, learning_rates_last=rates))
        if value < best_loss:
            best_loss = value
            best = dict(epoch=epoch, head=cpu(head), encoder=weights())
        save(dict(manifest_sha256=manifest_sha, size=size, budget=plan['recipe']['budget'], head=cpu(head), encoder=weights(),
            optimizer=optimizer.state_dict(), history=history, best=best, best_loss=best_loss, audit=audit,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)), resume)
        epoch_save(plan, size, model, audit, history, cpu(head), weights(), epoch, value, artifacts / f'epoch_{epoch:02d}.pt')
        write(out / 'history.json', history)
        print(f"{plan['job']} epoch {epoch}: XC validation BCE={value:.6f}; best epoch {best['epoch']}", flush=True)
    final = dict(detector_architecture=ARCHITECTURE, manifest_sha256=manifest_sha, head=best['head'],
        encoder=best['encoder'], backbone_id=model['backbone'], backbone_revision=model['revision'],
        seed=plan['seed'], hidden=0, head_layers=0, height=4, width=1000, dropout=.1, layer_norm=False,
        target_smoothing=False, tv_weight=0, training_config=plan['training_config'],
        training_recipe=dict(scheduled=False, layerwise=False, head_only_epochs=0), training_history=history,
        initial_head_sha256=audit['initial_head_sha256'], audit=audit,
        training_recording_ids=plan['datasets']['budgets'][str(plan['recipe']['budget'])]['recordings'],
        validation_recording_ids=plan['datasets']['validation']['recording_ids'],
        annotations_sha256='25000-pool-prefix',
        validation_annotations_sha256=plan['datasets']['validation']['sha256'],
        metrics=dict(epochs=plan['epochs'], selected_epoch=best['epoch'], best_val_loss=best_loss,
            head_only_epochs=0, full_finetuning_epochs=plan['epochs'],
            train_timebins=plan['recipe']['budget'] * 200,
            validation_timebins=2500 * 200,
            checkpoint_selection='minimum_recording_disjoint_validation_loss'))
    save(final, artifacts / 'model.pt')
    write(out / 'checkpoint.json', dict(path=str(artifacts / 'model.pt'), sha256=sha(artifacts / 'model.pt'),
        manifest_sha256=manifest_sha, metrics=final['metrics'], audit=audit))


def eval_best(plan, out, artifacts):
    import torch
    import evaluate_current_backbones as scorer
    from birdsong_detect_distill.full_encoder_linear import load_heads
    from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate
    anchor = read(SOURCE / 'anchor.json')
    reference = anchor['protocol']
    scorer.load_heads = load_heads
    best_epoch = read(out / 'checkpoint.json')['metrics']['selected_epoch']
    ckpt = artifacts / f'epoch_{best_epoch:02d}.pt'
    saved = torch.load(ckpt, map_location='cpu', weights_only=True)
    frozen(out / 'protocol.json', dict(manifest={k: v for k, v in plan.items() if k != 'out_manifest'},
        checkpoint={k: v for k, v in saved.items() if k not in ['encoder', 'head']},
        checkpoint_path=str(ckpt), checkpoint_sha256=sha(ckpt),
        anchor=str(SOURCE / 'anchor.json'), anchor_sha256=sha(SOURCE / 'anchor.json'),
        student_inference=read(SOURCE / 'inference.json')))
    if not (out / 'comparison.json').exists():
        scorer.score_checkpoint(ckpt, saved, out, artifacts / 'predictions',
            SOURCE / 'raw', reference, anchor, 'samples')
    if not (out / 'loro.json').exists():
        report = read(out / 'comparison.json')
        spec = dict(id=plan['job'], family='full_encoder_linear', label=plan['job'],
            seed=plan['seed'], threshold_floor_index=0, path=out / 'comparison.json')
        write(out / 'loro.json', cross_calibrate(spec, report, checked_rows(report)))
    torch.cuda.empty_cache()


def worker(job):
    import torch
    master = read(OUT / 'manifest.json')
    for relative, expected in master['code_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen code changed: ' + relative)
    out = OUT / 'runs' / job
    artifacts = ROOT / 'artifacts' / job
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    lock = (out / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = {**master, 'job': job, 'recipe': master['jobs'][job],
        'software': {k: version(k) for k in ['torch', 'transformers', 'numpy', 'scipy', 'librosa']}}
    plan['size'] = plan['recipe']['size']
    plan['seed'] = plan['recipe']['seed']
    plan['initialization'] = master['initializations'][f"{plan['size']}_s{plan['seed']}"]
    frozen(out / 'manifest.json', {k: v for k, v in plan.items() if k != 'out_manifest'})
    try:
        if not (artifacts / 'model.pt').exists():
            train_job(plan, out, artifacts)
        torch.cuda.empty_cache()
        eval_best(plan, out, artifacts)
        score = read(out / 'loro.json')['summary']
        hist = {h['epoch']: h for h in read(out / 'history.json')}
        best_epoch = read(out / 'checkpoint.json')['metrics']['selected_epoch']
        write(out / 'result.json', dict(job=job, size=plan['size'], budget=plan['recipe']['budget'], seed=plan['seed'],
            best_epoch=best_epoch, ap=score['ap'], iou=score['iou'],
            xc_val_bce=hist[best_epoch]['validation_loss'], history=read(out / 'history.json')))
        write(out / 'status.json', dict(state='complete', updated_unix=time.time()))
    except BaseException as error:
        write(out / 'status.json', dict(state='failed', error=repr(error), updated_unix=time.time()))
        raise


def lane(gpu):
    time.sleep(gpu * 5)
    for job in JOBS:
        status = OUT / 'runs' / job / 'status.json'
        if status.exists() and read(status)['state'] == 'complete':
            continue
        with (OUT / 'logs' / f'{job}.log').open('a') as log:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__)), '--job', job],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(f'lane {gpu}: {job} busy/failed, moving on', flush=True)


def driver():
    OUT.mkdir(exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    lock = (OUT / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        write(OUT / 'status.json', dict(state='preparing', updated_unix=time.time()))
        plan = prepare()
        write(OUT / 'status.json', dict(state='running', updated_unix=time.time()))
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(lane, gpu) for gpu in [0, 1]]
            for future in futures:
                future.result()
        rows = []
        for job, cfg in JOBS.items():
            out = OUT / 'runs' / job
            status = read(out / 'status.json')
            if status['state'] != 'complete':
                raise ValueError(f'Incomplete job: {job}')
            result = read(out / 'result.json')
            checkpoint = read(out / 'checkpoint.json')
            if sha(checkpoint['path']) != checkpoint['sha256']:
                raise ValueError('Checkpoint changed')
            rows.append(result)
        write(OUT / 'scaling.json', rows)
        lines = ['# Scaling seeds 1-2 — new default, 5 epochs, best XC BCE', '']
        for seed in SEEDS:
            lines += [f'## Seed {seed}', '',
                '| Budget (s) | Micro AP | Micro IoU | Base AP | Base IoU | Large AP | Large IoU |',
                '|---:|---:|---:|---:|---:|---:|---:|']
            by = {(r['size'], r['budget']): r for r in rows if r['seed'] == seed}
            for budget in BUDGETS:
                cells = [str(budget)]
                for size in SIZES:
                    r = by[(size, budget)]
                    cells += [f"{r['ap']:.4f}", f"{r['iou']:.4f}"]
                lines.append('| ' + ' | '.join(cells) + ' |')
            lines += ['']
        (OUT / 'scaling.md').write_text('\n'.join(lines) + '\n')
        write(OUT / 'status.json', dict(state='complete', updated_unix=time.time()))
    except BaseException as error:
        write(OUT / 'status.json', dict(state='failed', error=repr(error), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    if platform.node() != 'Lambda-Twins' or ROOT != Path('/mnt/birdconv/songmae_perch2/birdsong_detect_distill_experiments/scaling-default-seeds12-20260918'):
        raise SystemExit('Run only in the dedicated scaling directory')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=JOBS)
    args = parser.parse_args()
    os.chdir(ROOT)
    worker(args.job) if args.job else driver()
