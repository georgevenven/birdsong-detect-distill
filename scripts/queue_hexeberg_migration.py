"""Detached hardware migration and automatic evaluation, preserving the study inputs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import hexeberg_suite as s

REMOTE = '/media/george/DATA/yolo-hexeberg-nano-20260921'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o',
       'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=15 george-vengrovski@163.41.128.14 -W %h:%p',
       'george@george-server']
AUDIT = s.ROOT/'migration_20260921'


def call(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def transfer(source, destination):
    call(['rsync', '-a', '--partial', '-e', shlex.join(SSH[:-1]), str(source), str(destination)])


def status(state, **extra):
    s.atomic(AUDIT/'status.json', dict(state=state, updated=time.time(), **extra))
    s.atomic(s.OUT/'status.json', dict(state=state, coordinator='migration_20260921', updated_unix=time.time(), **extra))


def evaluate(job):
    out = s.ART/job
    report = s.OUT/'runs'/job
    s.atomic(report/'checkpoint.json', dict(path=str(out/'model.pt'), sha256=s.sha(out/'model.pt'),
        seed=int(job[-1]), selection='Ultralytics native validation fitness, best.pt',
        manifest_sha256=s.sha(s.OUT/'manifest.json'),
        train_args_sha256=s.sha(out/'fit/args.yaml'), history_sha256=s.sha(out/'fit/results.csv'),
        initialization_sha256=s.sha(s.ROOT/f'models/coco_yolo11{job.split("_")[1]}.pt'),
        migration_manifest_sha256=s.sha(AUDIT/'manifest.json'),
        hardware=s.read(out/'training_complete.json')))
    env={**os.environ, 'CUDA_VISIBLE_DEVICES':'0'}
    for dataset in ['powdermill', 'wabad', 'hawaii', 'xcsl', 'nips4bplus']:
        status('evaluating', job=job, dataset=dataset)
        with (s.OUT/'logs'/f'{job}_evaluation.log').open('a') as log:
            call([sys.executable, '-u', str(s.ROOT/'scripts/run_hexeberg_yolo.py'),
                  '--job', job, '--mode', dataset], env=env, stdout=log, stderr=subprocess.STDOUT)
    s.atomic(report/'status.json', dict(state='complete', updated_unix=time.time()))


def main():
    s.prepare()
    if not (AUDIT/'launched.json').exists():
        status('waiting_for_transfer_and_checkpoint')
        while True:
            state=subprocess.check_output(['systemctl','--user','show','birdsong-yolo-transfer-20260921.service',
                                           '-p','ActiveState','--value'],text=True).strip()
            if state=='failed':
                raise RuntimeError('Dataset transfer failed')
            last=s.ART/'yolo_l_25000_s0/fit/weights/last.pt'
            if state=='inactive' and last.exists():
                break
            time.sleep(20)
        transfer(s.ROOT/'scripts/verify_hexeberg_transfer.py',f'{SSH[-1]}:{REMOTE}/scripts/')
        call(SSH+[f'{REMOTE}/venv/bin/python {REMOTE}/scripts/verify_hexeberg_transfer.py {REMOTE}'])
        # Stop only this study. Completed-epoch checkpoints are retained.
        call(['systemctl','--user','stop','birdsong-yolo-hexeberg-20260920.service'])
        status('transferring_nano_checkpoint')
        transfer(str(s.ART/'yolo_n_25000_s0')+'/', f'{SSH[-1]}:{REMOTE}/artifacts/yolo_n_25000_s0/')
        transfer(s.OUT/'manifest.json',f'{SSH[-1]}:{REMOTE}/results/')
        transfer(s.ROOT/'models/coco_yolo11n.pt',f'{SSH[-1]}:{REMOTE}/models/')
        transfer(s.ROOT/'scripts/hexeberg_distributed.py',f'{SSH[-1]}:{REMOTE}/scripts/')
        call(SSH+[f'cp {REMOTE}/setup/dataset.yaml {REMOTE}/dataset/dataset.yaml'])
        call(SSH+[f'cp {REMOTE}/models/coco_yolo11n.pt {REMOTE}/yolo11n.pt'])
        # One single-GPU training process per seed; tmux survives disconnection.
        for seed in range(3):
            command=(f'cd {REMOTE} && env CUDA_VISIBLE_DEVICES={seed} OMP_NUM_THREADS=4 '
                f'OPENBLAS_NUM_THREADS=4 YOLO_CONFIG_DIR={REMOTE}/ultralytics_config '
                f'MPLCONFIGDIR={REMOTE}/matplotlib_cache '
                f'{REMOTE}/venv/bin/python -u scripts/hexeberg_distributed.py '
                f'--root {REMOTE} --variant n --seed {seed} > nano_seed{seed}.log 2>&1')
            call(SSH+['tmux new-session -d -s '+f'yolo-nano-{seed} '+shlex.quote(command)])
        s.atomic(AUDIT/'launched.json',dict(time=time.time()))
    # Large uses both GPUs, no host activation offload. Evaluate between seeds.
    for seed in range(3):
        job=f'yolo_l_25000_s{seed}'
        status('training_large',job=job)
        with (s.OUT/'logs'/f'{job}_ddp.log').open('a') as log:
            call([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                str(s.ROOT/'scripts/hexeberg_distributed.py'),'--root',str(s.ROOT),
                '--variant','l','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1'})
        evaluate(job)
    for seed in range(3):
        job=f'yolo_n_25000_s{seed}'
        status('waiting_for_nano',job=job)
        while True:
            check=call(SSH+[f'if test -f {REMOTE}/artifacts/{job}/training_complete.json; then echo complete; '
                f'elif test -f {REMOTE}/artifacts/{job}/training_failed.json; then echo failed; '
                f'elif tmux has-session -t yolo-nano-{seed} 2>/dev/null; then echo running; else echo stopped; fi'],
                capture_output=True,text=True).stdout.strip()
            if check=='complete':
                break
            if check!='running':
                raise RuntimeError(f'{job}: remote training {check}')
            time.sleep(60)
        transfer(f'{SSH[-1]}:{REMOTE}/artifacts/{job}/',str(s.ART/job)+'/')
        evaluate(job)
    rows=[]
    for variant in ['n','l']:
        for seed in range(3):
            job=f'yolo_{variant}_25000_s{seed}'
            for dataset in ['wabad','hawaii','xcsl','nips4bplus']:
                report=s.read(s.OUT/'external'/job/f'{dataset}.json')
                metric='full' if dataset in ['wabad','hawaii'] else 'temporal'
                score=report['site_macro' if dataset=='wabad' else 'summary'][metric]
                rows.append(dict(job=job,dataset=dataset,ap=score['ap'],iou=score['iou']))
    s.atomic(s.OUT/'external_summary.json',rows)
    status('complete')
    s.atomic(s.OUT/'status.json',dict(state='complete',jobs=6,updated_unix=time.time()))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        status('failed',error=repr(error))
        raise
