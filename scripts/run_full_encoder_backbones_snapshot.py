#!/usr/bin/env python3
"""Matched Micro/Base/Large full-encoder fine-tuning on a frozen completed XC pool."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

import evaluate_current_backbones as scorer
import qwen_prompt_study as teacher
import run_full_encoder_linear_2k as original
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.data import PixelWindows, load_spec_slice, read_rows
from birdsong_detect_distill.full_encoder_linear import ARCHITECTURE, encoder_key, encoder_state, restore_encoder, load_heads
from birdsong_detect_distill.model import load_backbone, loss
from birdsong_detect_distill.pointwise_bare_linear import BareLinearHead
from prepare_detector_study import verify
from run_qwen_xc_50k import validate as accepted
from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate
from train import RecordedSampler, evaluate as validation_loss

ROOT = Path(__file__).resolve().parents[1]
RUN = 'full_encoder_backbones_snapshot_2026-09-14'
OUT = ROOT / 'results/qwen_teacher_powdermill' / RUN
ARTIFACTS = original.ARTIFACTS.parent / RUN
LABELS = ROOT / 'data/annotations/xcl' / RUN
QUEUE = ROOT / 'data/annotations/xcl/powdermill_protocol_50000s_2026-09-14'
PARENT = ROOT / 'results/qwen_teacher_powdermill/new_teacher_2000train_800val_2026-09-14/manifest.json'


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen input changed: {path}')
    else:
        write_json(path, value)


def progress(out, state, **values):
    write_json(out / 'status.json', dict(state=state, updated_unix=time.time(), **values))


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text()); verify(plan)
        return plan
    parent = json.loads(PARENT.read_text()); verify(parent)
    queue = json.loads((QUEUE / 'manifest.json').read_text())
    queue_sha = digest(QUEUE / 'manifest.json')
    snap = OUT / 'snapshot.json'
    if not snap.exists():
        names = {p.stem for p in (QUEUE / 'annotations/self_review_1').glob('*.json')}
        frozen(snap, dict(created_unix=time.time(), queue_sha256=queue_sha,
            names=[w['name'] for w in queue['windows'] if w['name'] in names]))
    snapshot = json.loads(snap.read_text())
    if snapshot['queue_sha256'] != queue_sha:
        raise ValueError('annotation queue changed')
    selected = set(snapshot['names'])
    windows = [w for w in queue['windows'] if w['name'] in selected]
    validation = parent['datasets']['validation']
    val_ids = set(validation['recording_ids'])
    selection = json.loads((QUEUE / 'selection.json').read_text())
    excluded = set().union(*(set(v.get('xc_ids', [])) for v in selection['exclusions'].values()))
    protected = {str(p):digest(p) for p in [PARENT, snap, QUEUE / 'manifest.json',
        QUEUE / 'selection.json', Path(validation['path']), original.CONTROL]}
    rows, workers, training_windows = [], Counter(), []
    for w in windows:
        for stage in ['reasoning', 'self_review_1']:
            if not accepted(QUEUE, w, queue_sha, stage):
                raise ValueError('incomplete teacher stages: ' + w['name'])
            source = QUEUE / 'annotations' / stage / f'{w["name"]}.json'
            protected[str(source)] = digest(source)
        annotation = json.loads(source.read_text())
        if annotation['events'] != teacher.parsed_events(annotation['raw_annotation'], w):
            raise ValueError('teacher event conversion changed')
        workers[annotation['worker_role']] += 1
        name, shard, start, end, left, right = w['tile']
        if name in excluded or right-left != 1000 or not 0 <= left < right <= end-start:
            raise ValueError('excluded recording or invalid window')
        if name in val_ids:
            continue
        raw = load_spec_slice(Path(queue['spec_dir']) / 'shards' / shard, start+left, start+right)
        if hashlib.sha256(raw.tobytes()).hexdigest() != w['spectrogram_sha256']:
            raise ValueError('teacher/student spectrogram mismatch')
        rows.append(dict(status='ok', recording=name, source=dict(shard=shard,start=start,end=end),
            tile=dict(start_timebin=left,end_timebin=right,ownership_start_timebin=left,ownership_end_timebin=right),
            events=annotation['events'], annotation=str(source), annotation_sha256=digest(source),
            manifest_sha256=queue_sha, worker_role=annotation['worker_role']))
        training_windows.append(w)
    if (len(windows) != len(selected) or len({w['tile'][0] for w in windows}) != len(windows)
            or not val_ids <= {w['tile'][0] for w in windows}):
        raise ValueError('duplicate recordings or missing original validation windows')
    LABELS.mkdir(parents=True, exist_ok=True)
    labels = LABELS / 'train.jsonl'
    content = ''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows)
    if labels.exists() and labels.read_text() != content:
        raise ValueError('frozen training labels changed')
    if not labels.exists():
        labels.write_text(content)
    config = AutoConfig.from_pretrained(parent['backbone'],revision=parent['revision'],trust_remote_code=True,local_files_only=True)
    data = PixelWindows(read_rows(labels,0), ROOT / 'data/xcl/shards', config)
    inputs, masks = hashlib.sha256(), hashlib.sha256()
    for spec, target, valid in data:
        if valid != 1000 or not torch.isfinite(spec).all() or not torch.all((target==0)|(target==1)):
            raise ValueError('invalid tensor or nonbinary targets')
        inputs.update(spec.numpy().tobytes()); masks.update(target.numpy().tobytes())
    training = dict(path=str(labels),sha256=digest(labels),seconds=len(data)*5,windows=len(data),
        recording_ids=sorted(r['recording'] for r in rows),preflight_input_sha256=inputs.hexdigest(),
        preflight_mask_sha256=masks.hexdigest(),supervision=data.supervision_counts(),
        focal_species=len({w['metadata']['ebird_code'] for w in training_windows}),
        coordinate_sites=len({w['coordinate_site_0_01_degree'] for w in training_windows}))
    if (len(data) != len(rows) or set(training['recording_ids']) & val_ids
            or not set(parent['datasets']['new2k']['recording_ids']) <= set(training['recording_ids'])):
        raise ValueError('split overlap, changed budget or missing earlier training recordings')
    protected[str(labels)] = digest(labels)
    settings = json.loads((original.OUT / 'manifest.json').read_text())['training_config']
    codes = [*parent['code_sha256'], str(Path(__file__)), 'scripts/run_full_encoder_linear_2k.py',
        'scripts/run_last_block_linear_2k.py','scripts/summarize_powdermill_loro.py']
    codes += [str(p) for p in (ROOT/'src/birdsong_detect_distill').glob('*linear.py')]
    codes += [str(p) for p in (ROOT/'src/birdsong_detect_distill').glob('pointwise_*.py')]
    plan = dict(run=RUN,architecture=ARCHITECTURE,seed=0,epochs=5,training_config=settings,
        datasets=dict(train=training,validation=validation),models={size:dict(
            backbone=f'georgeven/songmae-{size}-32x1',revision=revision) for size,revision in scorer.REVISIONS.items()},
        selection=dict(snapshot=str(snap),windows=len(windows),seconds=len(windows)*5,workers=dict(workers),
            rule='all completed reasoning + one self-review windows at snapshot; retain original 800s validation',
            caveat='completed-pool availability can depend on annotation latency'),
        training='Original pretrained backbone, full detection-path fine-tuning (including CNN and positions), bare linear head; hard BCE, no soft targets or TV; backbone eval mode, FP32 accumulation.',
        checkpoint_selection='Minimum source-disjoint XC validation BCE within five epochs.',
        sampling='Reset torch CPU/CUDA RNG to seed 0 after model/head initialization, giving every size identical minibatch order.',
        evaluation='All 77 Powdermill segments, leave-one-original-recording-out threshold calibration; retain legacy Recording_1 report.',
        inference='Existing native frontend; 5s windows, 2.5s stride, maximum overlap, Gaussian probabilities sigma=(2,3), one calibrated threshold; continuous-score pixel AP.',
        caveat='Single seed; common recipe, not per-size hyperparameter tuning. Powdermill is development data. Historical 2k runs have a different budget and are not matched controls.',
        protected_files=protected,code_sha256={p:digest(p) for p in codes},artifacts=str(ARTIFACTS))
    verify(plan); frozen(path, plan)
    return plan


def train(plan, size, out, artifacts, checkpoint):
    manifest_sha = digest(OUT / 'manifest.json')
    if checkpoint.exists():
        record = json.loads((out/'checkpoint.json').read_text())
        if record['manifest_sha256'] != manifest_sha or digest(checkpoint) != record['sha256']:
            raise ValueError('completed checkpoint changed')
        return
    torch.manual_seed(plan['seed']); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0'); model = plan['models'][size]
    backbone = load_backbone(model['backbone'],device,model['revision']); config = backbone.config
    if (config.mels,config.num_timebins,config.patch_height,config.patch_width) != (128,1000,32,1):
        raise ValueError('unexpected model grid')
    for name,p in backbone.named_parameters():
        p.requires_grad_(encoder_key(name))
    head = BareLinearHead(config.enc_hidden_d,0,4,1000,32,1,dropout=.1,layers=0).to(device)
    data = original.previous.dataset(plan,'train',config)
    val = original.previous.dataset(plan,'validation',config)
    trainable = {n:p for n,p in backbone.named_parameters() if p.requires_grad}
    unused = lambda: {n:v for n,v in backbone.state_dict().items() if not encoder_key(n)}
    weights = lambda: {n:v.detach().cpu().clone() for n,v in encoder_state(backbone).items()}
    hash_state, cpu, save = original.state_hash, original.cpu_state, original.save_atomic
    initial = {n:hash_state({n:p}) for n,p in trainable.items()}
    audit = dict(initial_head_sha256=hash_state(head.state_dict()),initial_encoder_sha256=hash_state(encoder_state(backbone)),
        unused_pretraining_state_sha256=hash_state(unused()),encoder_parameters=sum(p.numel() for p in trainable.values()),
        head_parameters=sum(p.numel() for p in head.parameters()),trainable_backbone_names=list(trainable))
    settings = plan['training_config']
    optimizer = torch.optim.AdamW([{'params':head.parameters(),'lr':settings['learning_rate']},
        {'params':list(trainable.values()),'lr':settings['backbone_learning_rate']}],weight_decay=settings['weight_decay'])
    torch.manual_seed(plan['seed'])  # Head/backbone sizes consume different initialization RNG counts.
    sampler = RecordedSampler(data)
    loader = DataLoader(data,16,sampler=sampler,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val,4,num_workers=2,pin_memory=True)
    resume = artifacts/'resume.pt'; history, best, best_loss = [], None, float('inf')
    if resume.exists():
        saved = torch.load(resume,map_location='cpu',weights_only=True)
        if saved['manifest_sha256'] != manifest_sha or saved['size'] != size:
            raise ValueError('resume provenance mismatch')
        restore_encoder(backbone,saved['encoder']); head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        history,best,best_loss,audit = [saved[k] for k in ['history','best','best_loss','audit']]
        torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'],device)
        del saved
    print(size, 'trainable encoder/head:', audit['encoder_parameters'],audit['head_parameters'],flush=True)
    for epoch in range(len(history)+1,plan['epochs']+1):
        head.train(); losses=[]
        progress(out,'training',size=size,epoch=epoch,epochs=plan['epochs'],step=0,steps=len(loader))
        for step,(specs,targets,valid) in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True); total=int(valid.sum()); batch_loss=0.
            for start in range(0,len(specs),settings['microbatch_size']):
                section=slice(start,start+settings['microbatch_size']); weight=int(valid[section].sum())/total
                tokens=backbone(input_values=specs[section].to(device),valid_timebins=valid[section].to(device)).last_hidden_state
                value=loss(head(tokens.float(),valid[section]),targets[section].to(device),valid[section],smooth=False,tv_weight=0)
                if not torch.isfinite(value): raise ValueError('nonfinite training loss')
                (value*weight).backward(); batch_loss+=float(value.detach())*weight
                del tokens,value
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in [*trainable.values(),*head.parameters()]):
                raise ValueError('missing or nonfinite gradient')
            optimizer.step(); losses.append(batch_loss)
            if epoch==1 and step==1:
                if any(hash_state({n:p})==initial[n] for n,p in trainable.items()) or hash_state(head.state_dict())==audit['initial_head_sha256']:
                    raise ValueError('trainable parameters did not update')
                audit.update(all_encoder_parameters_updated=True,head_updated=True,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
                write_json(out/'initialization.json',audit)
            if step%5==0 or step==len(loader):
                progress(out,'training',size=size,epoch=epoch,epochs=plan['epochs'],step=step,steps=len(loader),training_loss=float(np.mean(losses)))
                print(f'{size} epoch {epoch} step {step}/{len(loader)} train={np.mean(losses):.4f}',flush=True)
        torch.cuda.empty_cache()
        value,_=validation_loss(backbone,head,val_loader,val,device,0,target_smoothing=False,collect_rows=False)
        if not np.isfinite(value): raise ValueError('nonfinite validation loss')
        history.append(dict(epoch=epoch,mean_training_loss=float(np.mean(losses)),validation_loss=value,
            sample_order_sha256=sampler.digest.hexdigest(),cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
        if value<best_loss:
            best_loss=value; best=dict(epoch=epoch,head=cpu(head),encoder=weights())
        save(dict(manifest_sha256=manifest_sha,size=size,head=cpu(head),encoder=weights(),optimizer=optimizer.state_dict(),
            history=history,best=best,best_loss=best_loss,audit=audit,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)),resume)
        write_json(out/'history.json',history)
        print(f'{size} epoch {epoch}: XC validation BCE={value:.6f}',flush=True)
    if hash_state(unused())!=audit['unused_pretraining_state_sha256']:
        raise ValueError('unused pretraining weights changed')
    audit['unused_pretraining_weights_unchanged']=True
    saved=dict(detector_architecture=ARCHITECTURE,manifest_sha256=manifest_sha,head=best['head'],encoder=best['encoder'],
        backbone_id=model['backbone'],backbone_revision=model['revision'],seed=plan['seed'],hidden=0,head_layers=0,
        height=4,width=1000,dropout=.1,layer_norm=False,target_smoothing=False,tv_weight=0,training_config=settings,
        training_history=history,initial_head_sha256=audit['initial_head_sha256'],audit=audit,
        training_recording_ids=plan['datasets']['train']['recording_ids'],validation_recording_ids=plan['datasets']['validation']['recording_ids'],
        annotations_sha256=plan['datasets']['train']['sha256'],validation_annotations_sha256=plan['datasets']['validation']['sha256'],
        metrics=dict(epochs=plan['epochs'],selected_epoch=best['epoch'],best_val_loss=best_loss,
            train_timebins=plan['datasets']['train']['seconds']*200,validation_timebins=plan['datasets']['validation']['seconds']*200,
            checkpoint_selection='minimum_recording_disjoint_validation_loss'))
    save(saved,checkpoint)
    write_json(out/'checkpoint.json',dict(path=str(checkpoint),sha256=digest(checkpoint),manifest_sha256=manifest_sha,metrics=saved['metrics'],audit=audit))


def evaluate(plan,size,out,artifacts,checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    control=json.loads(original.CONTROL.read_text()); anchor_path=Path(control['protocol']['anchor'])
    anchor=json.loads(anchor_path.read_text()); reference=anchor['protocol']
    for name,key in [('wav_Files.zip','audio_sha256'),('annotation_Files.zip','references_sha256')]:
        if digest(original.RAW/'powdermill'/name)!=reference[key]: raise ValueError('Powdermill inputs changed')
    frozen(out/'protocol.json',dict(manifest=plan,checkpoint={k:v for k,v in saved.items() if k not in ['encoder','head']},
        checkpoint_path=str(checkpoint),checkpoint_sha256=digest(checkpoint),anchor=str(anchor_path),
        anchor_sha256=digest(anchor_path),student_inference=control['protocol']['student_inference']))
    progress(out,'evaluating',size=size,total=77)
    scorer.load_heads=load_heads
    scorer.score_checkpoint(checkpoint,saved,out,artifacts/'predictions',original.RAW,reference,anchor,'samples')
    report=json.loads((out/'comparison.json').read_text()); rows=checked_rows(report)
    spec=dict(id=size,family='full_encoder_linear',label=f'SongMAE-{size} full encoder + linear',seed=0,
        threshold_floor_index=0,path=out/'comparison.json')
    write_json(out/'loro.json',cross_calibrate(spec,report,rows))


def worker(plan,size):
    out,artifacts=OUT/'runs'/size,ARTIFACTS/size
    out.mkdir(parents=True,exist_ok=True); artifacts.mkdir(parents=True,exist_ok=True)
    lock=(out/'driver.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        verify(plan)
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['state']=='complete': return
        if torch.cuda.mem_get_info(0)[0]<20*1024**3: raise ValueError('GPU occupied; preserve other work')
        checkpoint=artifacts/f'{size}_s0.pt'
        train(plan,size,out,artifacts,checkpoint)
        evaluate(plan,size,out,artifacts,checkpoint)
        verify(plan); progress(out,'complete',size=size,completed=77,total=77)
    except Exception as error:
        progress(out,'failed',size=size,error=repr(error)); raise


def run(plan):
    for suffix in ['twins','server']:
        state=subprocess.check_output(['systemctl','--user','show',f'birdsong-qwen-xc50k-{suffix}-20260914.service','-p','ActiveState','--value'],text=True).strip()
        if state!='inactive': raise ValueError('Twins Qwen is not paused; preserve work')
    if shutil.disk_usage(ARTIFACTS.parent).free<12*1024**3 or shutil.disk_usage(ROOT).free<1024**3:
        raise ValueError('insufficient output disk space')
    if any(torch.cuda.mem_get_info(i)[0]<20*1024**3 for i in [0,1]): raise ValueError('Twins GPUs occupied')
    def lane(gpu,sizes):
        for size in sizes:
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'4',
                'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
            with (OUT/f'{size}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'--size',size],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    progress(OUT,'running',training_seconds=plan['datasets']['train']['seconds'],validation_seconds=plan['datasets']['validation']['seconds'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,0,['large']),pool.submit(lane,1,['micro','base'])]
        for future in futures: future.result()
    histories={s:json.loads((OUT/'runs'/s/'history.json').read_text()) for s in plan['models']}
    orders={s:[(r['sample_order_sha256'],r['cpu_rng_sha256']) for r in h] for s,h in histories.items()}
    if any(order!=orders['large'] for order in orders.values()): raise ValueError('minibatch order or RNG differs across sizes')
    results={s:dict(**json.loads((OUT/'runs'/s/'loro.json').read_text())['summary'],
        selected_epoch=min(h,key=lambda r:r['validation_loss'])['epoch']) for s,h in histories.items()}
    write_json(OUT/'summary.json',dict(training_seconds=plan['datasets']['train']['seconds'],validation_seconds=800,
        seed=0,matched_sample_order=True,models=results,caveat=plan['caveat']))
    progress(OUT,'complete',summary=str(OUT/'summary.json')); print(json.dumps(results,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--size',choices=scorer.REVISIONS)
    args=parser.parse_args(); os.chdir(ROOT); OUT.mkdir(parents=True,exist_ok=True)
    if args.size:
        plan=json.loads((OUT/'manifest.json').read_text()); worker(plan,args.size)
        return
    lock=(OUT/'driver.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        plan=prepare()
        print(json.dumps(dict(train_seconds=plan['datasets']['train']['seconds'],validation_seconds=plan['datasets']['validation']['seconds'],snapshot=plan['selection']),indent=2),flush=True)
        if not args.prepare_only: run(plan)
    except Exception as error:
        progress(OUT,'failed',error=repr(error)); raise


if __name__=='__main__':
    main()
