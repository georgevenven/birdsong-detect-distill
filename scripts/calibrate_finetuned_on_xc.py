#!/usr/bin/env python3
"""Select thresholds on XC teacher validation masks; reuse unchanged Powdermill scores."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import run_full_encoder_backbones_snapshot as trained
from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.benchmark_metrics import THRESHOLDS, area, calibrate, summarize
from birdsong_detect_distill.data import load_spec_slice
from birdsong_detect_distill.full_encoder_linear import load_heads
from birdsong_detect_distill.model import load_backbone
from evaluate_songmae_smoothing import smooth
from plot_bare_linear_xc import predict
from prepare_detector_study import verify
from summarize_powdermill_loro import checked_rows, group

OUT = trained.OUT.parent / 'xc_threshold_calibration_full_encoder_3205s_2026-09-14'
ARTIFACTS = trained.ARTIFACTS.parent / OUT.name


def run(size):
    out, cache = OUT/size, ARTIFACTS/size
    out.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)
    study = json.loads((trained.OUT/'manifest.json').read_text()); verify(study)
    report_path = trained.OUT/'runs'/size/'comparison.json'
    report = json.loads(report_path.read_text()); saved = report['protocol']['checkpoint']
    checkpoint = Path(report['protocol']['checkpoint_path'])
    if digest(checkpoint) != report['protocol']['checkpoint_sha256']:
        raise ValueError('trained checkpoint changed')
    paths = [Path(__file__),report_path,checkpoint,trained.OUT/'manifest.json',
        trained.ROOT/'scripts/plot_bare_linear_xc.py',trained.ROOT/'scripts/calibrate_finetuned_on_xc.py',
        trained.ROOT/'scripts/summarize_powdermill_loro.py',trained.OUT/'runs'/size/'loro.json']
    plan = dict(size=size,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        validation=study['datasets']['validation'],
        selection='Maximum mean 2D IoU across all 160 XC validation windows, including empty targets; first threshold on ties.',
        threshold_grid=THRESHOLDS.tolist(),label_source='Qwen reasoning plus one self-review; hard binary box-union masks, not human references.',
        inference='Full native source spectrogram, training normalization, 5s windows / 2.5s stride, maximum overlap, FP16 autocast; full-source Gaussian probability sigma=(2,3) before cropping.',
        coverage='Only the existing annotated 800 seconds are scored; other source context is not labeled background.',
        comparison='Frozen XC threshold applied to existing full-Powdermill probability threshold curves; no retraining, new Powdermill inference, or test-based threshold tuning.',
        caveat='The same XC validation set already selected training epoch. Powdermill remains development data because prior architecture and postprocessing decisions used it.',
        protected_files={**study['protected_files'],**{str(p):digest(p) for p in paths}},code_sha256=study['code_sha256'])
    trained.frozen(out/'manifest.json',plan); verify(plan)
    if (out/'comparison.json').exists():
        print(size,'already complete'); return
    if torch.cuda.mem_get_info(0)[0] < 20*1024**3:
        raise ValueError('GPU occupied; preserve existing jobs')
    device = torch.device('cuda:0')
    backbone = load_backbone(saved['backbone_id'],device,saved['backbone_revision'])
    head = load_heads([checkpoint],backbone,device)[checkpoint.stem]
    data = trained.original.previous.dataset(study,'validation',backbone.config)
    if (len(data)!=160 or set(saved['training_recording_ids']) & set(plan['validation']['recording_ids'])
            or saved['validation_annotations_sha256'] != plan['validation']['sha256']):
        raise ValueError('unexpected validation split')
    manifest_sha = digest(out/'manifest.json'); scores=[]
    for index,(row,start,valid,target) in enumerate(data.windows,1):
        name=row['recording']; source=row['source']; path=cache/f'{name}.npz'; result=out/'windows'/f'{name}.json'
        if result.exists():
            record=json.loads(result.read_text())
            if record['manifest_sha256']!=manifest_sha or digest(path)!=record['prediction_sha256']:
                raise ValueError('changed cached calibration prediction')
        else:
            raw=load_spec_slice(trained.ROOT/'data/xcl/shards'/Path(source['shard']).name,source['start'],source['end'])
            probability=predict(raw,backbone,head,device)
            values=smooth(probability)[:,start:start+valid]
            truth=target[:,:valid].astype(bool)
            if values.shape!=truth.shape or valid!=1000: raise ValueError('wrong calibration interval')
            np.savez_compressed(path,smoothed_probability=values)
            score=dict(name=name,group=name,seconds=valid/200,area={'full':area(values,truth)})
            record=dict(manifest_sha256=manifest_sha,score=score,prediction_sha256=digest(path),
                source_spectrogram_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                reference_mask_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),source=source,start_timebin=start,end_timebin=start+valid)
            write_json(result,record)
        scores.append(record['score'])
        if index%10==0 or index==1:
            write_json(out/'status.json',dict(state='calibrating',completed=index,total=len(data),updated_unix=time.time()))
            print(f'{size}: XC validation {index}/{len(data)}',flush=True)
    thresholds=calibrate(scores)
    calibration=dict(threshold_indices=thresholds,threshold=float(THRESHOLDS[thresholds['full']]),
        summary=summarize(scores,thresholds)['full'],mean_iou_curve=np.mean([r['area']['full']['iou_curve'] for r in scores],axis=0).tolist(),
        windows=len(scores),seconds=sum(r['seconds'] for r in scores),per_window=scores,selection=plan['selection'])
    # Freeze the XC decision before reading the Powdermill reference confusion counts.
    trained.frozen(out/'calibration.json',calibration)
    rows=checked_rows(report); baseline=json.loads((trained.OUT/'runs'/size/'loro.json').read_text())
    full=summarize(rows,thresholds)['full']
    counts=np.sum([r['area']['full']['counts'][thresholds['full']] for r in rows],axis=0)
    tp,fp,fn=counts
    full.update(pooled_iou=float(tp/max(1,tp+fp+fn)),segments=len(rows),seconds=sum(r['seconds'] for r in rows))
    if full['ap']!=baseline['summary']['ap']:
        raise ValueError('threshold-independent Powdermill AP changed')
    per_group={g:summarize([r for r in rows if group(r)==g],thresholds)['full'] for g in sorted({group(r) for r in rows})}
    compared=dict(size=size,training_seconds=3205,validation_seconds=800,selected_epoch=saved['metrics']['selected_epoch'],
        xc_threshold=calibration['threshold'],xc_calibration=calibration['summary'],powdermill=full,
        powdermill_by_original_recording=per_group,previous_powdermill_loro=baseline['summary'],
        previous_powdermill_thresholds={f['held_out']:f['threshold'] for f in baseline['folds']},
        iou_delta=full['iou']-baseline['summary']['iou'],ap_unchanged=True,manifest_sha256=manifest_sha,caveat=plan['caveat'])
    verify(plan); write_json(out/'comparison.json',compared)
    write_json(out/'status.json',dict(state='complete',completed=160,total=160,updated_unix=time.time()))
    print(json.dumps({k:compared[k] for k in ['size','xc_threshold','powdermill','iou_delta']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',nargs='+',choices=['micro','base','large'],required=True)
    args=parser.parse_args(); os.chdir(trained.ROOT)
    for size in args.sizes: run(size)


if __name__=='__main__': main()
