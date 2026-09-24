"""Additional detached V100 evaluation lane; dataset locks prevent duplicate work."""
import argparse
import os
import subprocess
import sys
import time
import hexeberg_suite as s
from queue_hexeberg_migration import AUDIT

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu',type=int,required=True)
parser.add_argument('--seed',type=int,required=True)
args=parser.parse_args()
job=f'yolo_n_25000_s{args.seed}'
try:
    for dataset in ['wabad','hawaii','xcsl','nips4bplus']:
        s.atomic(AUDIT/f'eval_gpu{args.gpu}.json',dict(state='evaluating',job=job,dataset=dataset,updated=time.time()))
        subprocess.run([sys.executable,'-u',str(s.ROOT/'scripts/evaluate_hexeberg_remote.py'),
            '--job',job,'--dataset',dataset],check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','YOLO_EVAL_GPU':str(args.gpu)})
    s.atomic(AUDIT/f'eval_gpu{args.gpu}.json',dict(state='complete',job=job,updated=time.time()))
except BaseException as error:
    s.atomic(AUDIT/f'eval_gpu{args.gpu}.json',dict(state='failed',job=job,error=repr(error),updated=time.time()))
    raise
