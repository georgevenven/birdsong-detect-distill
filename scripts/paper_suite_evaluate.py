#!/usr/bin/env python3
"""Native inference, common area metrics, frozen development thresholds; resume per recording."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import paper_suite as suite
from birdsong_detect_distill import hawaii
from birdsong_detect_distill.baseline_models import Predictor, maps
from birdsong_detect_distill.benchmark_data import digest, model_audio, read_audio, recordings
from birdsong_detect_distill.benchmark_metrics import RATE, area_scores, calibrate, intervals_mask, summarize
from birdsong_detect_distill.full_encoder_linear import load_heads
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import songmae_probabilities
from evaluate_songmae_smoothing import smooth
from merge_external_shards import by_site
from prepare_detector_study import verify


def identity(model, plan):
    if model=='songmae':
        checkpoint = suite.ART/'songmae/large_15000/model.pt'
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        for key,expected in [('annotations_sha256',plan['datasets']['15000']['sha256']),
            ('validation_annotations_sha256',plan['datasets']['validation']['sha256'])]:
            if saved[key]!=expected:
                raise ValueError('SongMAE training split differs')
        if saved['metrics']['train_timebins']!=15000*RATE or saved['target_smoothing'] or saved['tv_weight']:
            raise ValueError('wrong SongMAE recipe')
        metadata = dict(model=model,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
            backbone_id=saved['backbone_id'],backbone_revision=saved['backbone_revision'],
            detector_architecture=saved['detector_architecture'],training=saved['metrics'],
            inference='Full fine-tuned encoder + linear, FP16 autocast, native 128-mel frontend, 5s/2.5s, maximum overlap; probability Gaussian sigma=(2,3) before frequency collapse.')
        device = torch.device('cuda:0')
        torch.set_float32_matmul_precision('high')
        backbone = load_backbone(saved['backbone_id'],device,saved['backbone_revision'])
        heads = load_heads([checkpoint],backbone,device)
        def predict(audio, rate):
            return dict(probability=smooth(songmae_probabilities(backbone,heads,audio,device)[checkpoint.stem]))
        return 'songmae',metadata,predict
    kind = 'birdcode' if model=='birdcode' else 'birdbox' if model.startswith('released') else 'qwen_yolo'
    variant = 'yolo11'+model[-1] if kind!='birdcode' else 'yolo11n'
    checkpoint = suite.ART/'yolo'/model[-1]/'model.pt' if kind=='qwen_yolo' else None
    predictor = Predictor(kind,checkpoint,'cuda:0',batch_size=4 if kind=='birdcode' else 8,
        confidence=.00001,max_det=10000,variant=variant)
    if kind=='qwen_yolo':
        sidecar = predictor.metadata['training']
        if (sidecar['manifest_sha256']!=digest(suite.OUT/'manifest.json')
                or sidecar['sha256']!=predictor.metadata['checkpoint_sha256']):
            raise ValueError('YOLO training provenance differs')
    return kind,predictor.metadata,predictor.predict


def run(model, dataset):
    plan = suite.prepare()
    cache = suite.ART/'evaluation'/model
    out = suite.OUT/'external'/model
    cache.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/f'{dataset}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    destination = out/f'{dataset}.json'
    manifest_sha = digest(suite.OUT/'manifest.json')
    if destination.exists():
        report = json.loads(destination.read_text())
        if report['manifest_sha256']!=manifest_sha:
            raise ValueError('existing report belongs to another suite')
        verify(plan)
        return
    root = suite.ROOT/'data/hawaii/zenodo' if dataset=='hawaii' else suite.ROOT/'data/nips4bplus' if dataset=='nips4bplus' else suite.RAW
    inventory = {r['name']:r for r in (hawaii.recordings(root) if dataset=='hawaii' else recordings(root,dataset))}
    partitions = plan['powdermill'] if dataset=='powdermill' else {'evaluation':plan['external'][dataset]['recordings']}
    names = sorted(n for selected in partitions.values() for n in selected)
    if len(names)!=len(set(names)) or set(names)-inventory.keys():
        raise ValueError('incomplete or duplicate dataset coverage')
    if dataset=='powdermill' and (len(names)!=77 or {n.split('_Segment_')[0] for n in partitions['calibration']}!={'Recording_2','Recording_3','Recording_4'}):
        raise ValueError('Powdermill calibration groups differ')
    reference = json.loads(Path(plan['external'][dataset]['reference_report']).read_text()) if dataset!='powdermill' else None
    reference_rows = {r['name']:r for r in reference['per_recording']['evaluation']} if reference else {}
    reference_sources = {str(Path(p).resolve()):sha for p,sha in reference['source_audio_sha256'].items()} if reference else {}
    kind, metadata, predict = identity(model,plan)
    protocol = dict(manifest_sha256=manifest_sha,inference=metadata,
        metric_code_sha256=digest(suite.ROOT/'src/birdsong_detect_distill/benchmark_metrics.py'),
        evaluator_sha256=digest(Path(__file__)),grid='128 Slaney-mel bins 20–16000 Hz; 5ms frames',
        postprocessing='SongMAE probability Gaussian sigma=(2,3); no smoothing for native baseline scores; one calibrated threshold, no morphology.',
        metrics='Exact continuous-score pixel/frame AP, mean area IoU, pooled precision/recall; WABAD primary equal-site macro.',
        references='Frozen external exclusions and ignored intervals; zero-extent boxes removed before rasterization.')
    suite.frozen(cache/'protocol.json',protocol)
    protocol_sha = digest(cache/'protocol.json')
    calibration = json.loads((out/'powdermill.json').read_text()) if dataset!='powdermill' else None
    if calibration and (calibration['protocol_sha256']!=protocol_sha or calibration['diagnostic_only']):
        raise ValueError('calibration uses another model/protocol or incomplete data')
    thresholds = calibration['threshold_indices'] if calibration else None
    scored, sources = {}, {}
    samples = {names[0],names[len(names)//2],names[-1]}
    for partition, selected in partitions.items():
        scored[partition] = []
        for index,name in enumerate(sorted(selected),1):
            record = inventory[name]
            source_path = record.get('archive',record.get('path'))
            if source_path not in sources:
                sources[source_path] = digest(source_path)
            sha = sources[source_path]
            if record.get('expected_audio_sha256',sha)!=sha or reference and reference_sources[str(Path(source_path).resolve())]!=sha:
                raise ValueError('audio differs from frozen release')
            signature = dict(audio_sha256=sha,member=record.get('member'),
                reference_sha256=hashlib.sha256(record['events'].tobytes()).hexdigest(),ignored=record.get('ignored',[]))
            stem = name.replace('/','__')
            score_path = cache/'scores'/dataset/f'{stem}.json'
            if score_path.exists():
                result = json.loads(score_path.read_text())
                if result['source']!=signature or result['protocol_sha256']!=protocol_sha:
                    raise ValueError('changed cached score provenance')
                row = result['row']
            else:
                audio, rate = read_audio(record)
                duration = len(audio)/rate
                audio, rate = model_audio(audio,rate,kind)
                prediction = predict(audio,rate)
                array_hash = hashlib.sha256()
                for key,value in sorted(prediction.items()):
                    array_hash.update(key.encode())
                    array_hash.update(np.ascontiguousarray(value).tobytes())
                prediction_path = None
                if kind in ['birdbox','qwen_yolo'] or name in samples:
                    p = cache/'predictions'/dataset/f'{stem}.npz'
                    p.parent.mkdir(parents=True,exist_ok=True)
                    pending = p.with_suffix('.tmp.npz')
                    np.savez_compressed(pending,**prediction)
                    pending.replace(p)
                    prediction_path = str(p)
                probability, scores = maps(prediction,duration)
                events = record['events']
                if ((events[:,1]<events[:,0]) | (events[:,3]<events[:,2])).any():
                    raise ValueError('inverted reference bounds in retained recording')
                events = events[(events[:,1]>events[:,0]) & (events[:,3]>events[:,2])]
                ignored = record.get('ignored',[])
                row = dict(name=name,group=record['group'],seconds=float(intervals_mask([[0,duration]],len(scores),ignored).sum()/RATE),
                    area=area_scores(probability,scores,events,len(scores),[[0,duration]],ignored,record.get('temporal_only',False)))
                suite.atomic(score_path,dict(protocol_sha256=protocol_sha,source=signature,duration=duration,row=row,
                    prediction_array_sha256=array_hash.hexdigest(),prediction_path=prediction_path,
                    prediction_sha256=digest(prediction_path) if prediction_path else None))
            if reference:
                expected = reference_rows[name]
                if row['seconds']!=expected['seconds']:
                    raise ValueError('external scoring coverage differs')
                for metric,value in row['area'].items():
                    if value['counts'][0]!=expected['area'][metric]['counts'][0]:
                        raise ValueError(f'external reference mask differs: {name}/{metric}')
            scored[partition].append(row)
            suite.atomic(out/f'{dataset}.status.json',dict(state='evaluating',partition=partition,completed=index,total=len(selected)))
            print(f'{model} {dataset} {partition} {index}/{len(selected)} {name}',flush=True)
        if partition=='calibration':
            thresholds = calibrate(scored[partition])
            suite.atomic(out/'thresholds.json',dict(threshold_indices=thresholds,recordings=sorted(selected),
                protocol_sha256=protocol_sha,selection=plan['calibration']))
    report = dict(dataset=dataset,model=model,manifest_sha256=manifest_sha,protocol_sha256=protocol_sha,
        protocol=protocol,diagnostic_only=False,threshold_indices=thresholds,
        calibration_sha256=digest(out/'powdermill.json') if calibration else None,
        segments=len(scored['evaluation']),evaluated_seconds=sum(r['seconds'] for r in scored['evaluation']),
        summary=summarize(scored['evaluation'],thresholds),per_recording=scored,source_audio_sha256=sources)
    if dataset in ['wabad','hawaii']:
        report['per_site'],report['site_macro'] = by_site(scored['evaluation'],thresholds)
    if dataset=='powdermill' and model=='songmae':
        earlier = json.loads((suite.OUT/'runs/large_15000/comparison.json').read_text())
        check = {r['name']:r['area']['full'] for rows in earlier['per_recording'].values() for r in rows}
        if any(r['area']['full']!=check[r['name']] for rows in scored.values() for r in rows):
            raise ValueError('external evaluator does not reproduce SongMAE development pixel scores')
    verify(plan)
    suite.atomic(destination,report)
    suite.atomic(out/f'{dataset}.status.json',dict(state='complete',segments=report['segments']))
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=suite.MODELS,required=True)
    parser.add_argument('--dataset',choices=['powdermill','wabad','hawaii','xcsl','nips4bplus'],required=True)
    args = parser.parse_args()
    run(args.model,args.dataset)
