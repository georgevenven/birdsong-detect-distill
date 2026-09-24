"""Detached evaluation queue: CPU metrics on Twins, all inference on V100 GPU 2."""
import os
import subprocess
import sys
import time
import hexeberg_suite as s
from queue_hexeberg_migration import transfer, SSH, REMOTE, AUDIT


def status(state, **extra):
    value=dict(state=state,updated_unix=time.time(),coordinator='v100_gpu2',**extra)
    s.atomic(AUDIT/'v100_evaluation_status.json',value)
    s.atomic(s.OUT/'status.json',value)


def register(job):
    out=s.ART/job
    record=dict(path=str(out/'model.pt'),sha256=s.sha(out/'model.pt'),seed=int(job[-1]),
        selection='Ultralytics native validation fitness, best.pt',
        manifest_sha256=s.sha(s.OUT/'manifest.json'),
        train_args_sha256=s.sha(out/'fit/args.yaml'),history_sha256=s.sha(out/'fit/results.csv'),
        initialization_sha256=s.sha(s.ROOT/f'models/coco_yolo11{job.split("_")[1]}.pt'),
        migration_manifest_sha256=s.sha(AUDIT/'manifest.json'),hardware=s.read(out/'training_complete.json'),
        evaluation_execution=dict(host='george-server',gpu=2,transport='SSH framed NPZ waveforms',
            torch='2.5.1+cu121',worker_sha256=s.sha(s.ROOT/'scripts/hexeberg_remote_inference.py'),
            frontend_sha256=s.sha(s.ROOT/'src/birdsong_detect_distill/birdbox.py')))
    s.frozen(s.OUT/'runs'/job/'checkpoint.json',record)


def evaluate(job,dataset):
    status('evaluating',job=job,dataset=dataset)
    with (s.OUT/'logs'/f'{job}_v100_evaluation.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(s.ROOT/'scripts/evaluate_hexeberg_remote.py'),
            '--job',job,'--dataset',dataset],check=True,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''})


def main():
    s.prepare()
    # Get preliminary Powdermill numbers first, then the held-out datasets.
    for seed in range(3):
        job=f'yolo_n_25000_s{seed}'
        transfer(f'{SSH[-1]}:{REMOTE}/artifacts/{job}/',str(s.ART/job)+'/')
        register(job)
        evaluate(job,'powdermill')
    for seed in range(3):
        job=f'yolo_n_25000_s{seed}'
        for dataset in ['wabad','hawaii','xcsl','nips4bplus']:
            evaluate(job,dataset)
        s.atomic(s.OUT/'runs'/job/'status.json',dict(state='complete',updated_unix=time.time()))
    for seed in [1,2]:
        job=f'yolo_l_25000_s{seed}'
        if (s.ART/job/'local_evaluation.json').exists():
            status('waiting_for_local_evaluation',job=job)
            while True:
                progress=s.OUT/'runs'/job/'local_evaluation_status.json'
                value=s.read(progress) if progress.exists() else {}
                if value.get('state')=='complete':
                    break
                if value.get('state')=='failed':
                    raise RuntimeError('Local evaluation failed: '+job)
                time.sleep(60)
            continue
        status('waiting_for_training',job=job)
        if seed==1:
            while not (s.ART/job/'training_complete.json').exists():
                if (s.ART/job/'training_failed.json').exists():
                    raise RuntimeError('Large seed1 failed')
                time.sleep(60)
            # The old coordinator was paused, not its training subprocess.
            subprocess.run(['systemctl','--user','stop','birdsong-yolo-migration-20260921.service'],check=True)
            transfer(str(s.ART/job)+'/',f'{SSH[-1]}:{REMOTE}/artifacts/{job}/')
        else:
            subprocess.run([sys.executable,str(s.ROOT/'scripts/await_hexeberg_remote.py'),'--job',job],check=True)
        register(job)
        for dataset in ['powdermill','wabad','hawaii','xcsl','nips4bplus']:
            evaluate(job,dataset)
        s.atomic(s.OUT/'runs'/job/'status.json',dict(state='complete',updated_unix=time.time()))
    rows=[]
    for variant in ['n','l']:
        for seed in range(3):
            job=f'yolo_{variant}_25000_s{seed}'
            for dataset in ['wabad','hawaii','xcsl','nips4bplus']:
                report=s.read(s.OUT/'external'/job/f'{dataset}.json')
                score=report['site_macro' if dataset=='wabad' else 'summary']['full' if dataset in ['wabad','hawaii'] else 'temporal']
                rows.append(dict(job=job,dataset=dataset,ap=score['ap'],iou=score['iou']))
    s.atomic(s.OUT/'external_summary.json',rows)
    status('complete')


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        status('failed',error=repr(error))
        raise
