#!/usr/bin/env python3
"""Apply the existing native-input, resumable evaluator to each new training seed."""
import argparse
import json

import torch

import paper_25k_suite as suite
import paper_suite_evaluate as shared
from birdsong_detect_distill.baseline_models import Predictor
from birdsong_detect_distill.benchmark_data import digest
from birdsong_detect_distill.full_encoder_linear import load_heads
from birdsong_detect_distill.model import load_backbone
from evaluate_2d import songmae_probabilities
from evaluate_songmae_smoothing import smooth
from summarize_powdermill_loro import checked_rows, evaluate as cross_calibrate

released_identity = shared.identity


def identity(job, plan):
    if job in suite.RELEASED:
        kind, metadata, predict = released_identity(job,plan)
        metadata['seed_adapter_sha256'] = digest(__file__)
        return kind, metadata, predict
    family, size, budget, seed = suite.job_parts(job)
    checkpoint = suite.ART/family/job/'model.pt'
    if family == 'yolo':
        predictor = Predictor('qwen_yolo',checkpoint,'cuda:0',batch_size=8,
            confidence=.00001,max_det=10000,variant='yolo11'+size)
        metadata = predictor.metadata
        saved = metadata['training']
        if (saved['manifest_sha256'] != digest(suite.OUT/'manifest.json') or saved['seed'] != seed
                or saved['sha256'] != digest(checkpoint) or saved['epochs'] != 10
                or saved['dataset']['annotations_sha256'] != plan['datasets'][str(budget)]['sha256']
                or saved['dataset']['validation_annotations_sha256'] != plan['datasets']['validation']['sha256']):
            raise ValueError('YOLO training provenance differs')
        metadata['seed_adapter_sha256'] = digest(__file__)
        return 'qwen_yolo',metadata,predictor.predict
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if (saved['seed'] != seed or saved['metrics']['epochs'] != 10
            or saved['metrics']['train_timebins'] != budget*200 or saved['metrics']['validation_timebins'] != 2500*200
            or saved['annotations_sha256'] != plan['datasets'][str(budget)]['sha256']
            or saved['validation_annotations_sha256'] != plan['datasets']['validation']['sha256']
            or saved['target_smoothing'] or saved['tv_weight']):
        raise ValueError('SongMAE training provenance differs')
    checkpoint_report = json.loads((suite.OUT/'runs'/job/'checkpoint.json').read_text())
    if checkpoint_report['sha256'] != digest(checkpoint):
        raise ValueError('checkpoint changed')
    metadata = dict(model=job,seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        seed_adapter_sha256=digest(__file__),backbone_id=saved['backbone_id'],backbone_revision=saved['backbone_revision'],
        detector_architecture=saved['detector_architecture'],training=saved['metrics'],
        inference='Full fine-tuned encoder + linear; FP16 autocast; native 128 mel, 5s/2.5s maximum overlap; probability Gaussian sigma=(2,3).')
    device = torch.device('cuda:0'); torch.set_float32_matmul_precision('high')
    backbone = load_backbone(saved['backbone_id'],device,saved['backbone_revision'])
    heads = load_heads([checkpoint],backbone,device)
    def predict(audio,rate):
        return dict(probability=smooth(songmae_probabilities(backbone,heads,audio,device)[checkpoint.stem]))
    return 'songmae',metadata,predict


def run(job,dataset):
    shared.suite = suite
    shared.identity = identity
    shared.run(job,dataset)
    if dataset != 'powdermill' or job == 'birdcode':
        return
    path = suite.OUT/'external'/job/'powdermill.json'
    report = json.loads(path.read_text())
    rows = [dict(row,area={'full':row['area']['full']}) for row in checked_rows(report)]
    if job.startswith(('micro_', 'base_', 'large_')):
        earlier = json.loads((suite.OUT/'runs'/job/'comparison.json').read_text())
        reference = {r['name']:r['area']['full'] for r in checked_rows(earlier)}
        if any(r['area']['full'] != reference[r['name']] for r in rows):
            raise ValueError('external scoring does not reproduce development pixel scores')
    seed = suite.job_parts(job)[3] if job not in suite.RELEASED else 0
    spec = dict(id=job,family=job.rsplit('_s',1)[0],label=job,seed=seed,threshold_floor_index=0,path=path)
    result = cross_calibrate(spec,dict(report,summary=report['summary']['full']),rows)
    suite.atomic(path.with_name('loro.json'),result)
    if job.startswith('yolo_'):
        suite.atomic(suite.OUT/'runs'/job/'status.json',dict(state='complete',job=job,seed=seed,
            powdermill_segments=77,checkpoint_selection='best XC validation box mAP50-95'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=[*suite.SONGMAE_JOBS,*suite.YOLO_JOBS,*suite.RELEASED],required=True)
    parser.add_argument('--dataset',choices=['powdermill','xcsl','nips4bplus','wabad','hawaii'],required=True)
    args = parser.parse_args()
    run(args.model,args.dataset)
