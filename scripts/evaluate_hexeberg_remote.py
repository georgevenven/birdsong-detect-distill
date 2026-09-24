"""Canonical CPU scoring on Twins with native YOLO inference on V100 GPU 2."""
import argparse
import fcntl
import os
import shlex
import subprocess
import time
import hexeberg_suite as s
from queue_hexeberg_migration import SSH, REMOTE
from hexeberg_remote_inference import send, receive

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--job',required=True)
parser.add_argument('--dataset',required=True)
args=parser.parse_args()
folder=s.OUT/'external'/args.job
folder.mkdir(parents=True,exist_ok=True)
lock=(folder/f'{args.dataset}.dispatch.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
gpu=int(os.getenv('YOLO_EVAL_GPU','2'))
worker_name='hexeberg_remote_inference.py' if gpu==2 else 'hexeberg_remote_inference_multigpu.py'
checkpoint=s.read(s.OUT/'runs'/args.job/'checkpoint.json')
command=(f'cd {REMOTE} && env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
         f'PYTHONPATH={REMOTE}/eval_src YOLO_CONFIG_DIR={REMOTE}/ultralytics_config '
         f'MPLCONFIGDIR={REMOTE}/matplotlib_cache {REMOTE}/venv/bin/python -u '
         f'scripts/{worker_name} --checkpoint {REMOTE}/artifacts/{args.job}/model.pt '
         f'--sha256 {shlex.quote(checkpoint["sha256"])}'+(f' --gpu {gpu}' if gpu!=2 else ''))
if not (folder/f'{args.dataset}.json').exists():
    s.atomic(folder/f'{args.dataset}.execution.json',dict(host='george-server',gpu=gpu,
        worker_sha256=s.sha(s.ROOT/'scripts'/worker_name),note='Same inference settings; physical GPU routing only.'))
def start_worker():
    return subprocess.Popen(SSH[:-1]+['-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',SSH[-1],command],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE)

def stop_worker():
    try:
        worker.stdin.close()
    except (BrokenPipeError,OSError):
        pass
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill();worker.wait()

worker=start_worker()
def predict(model,audio,rate,device,**kwargs):
    global worker
    for attempt in range(3):
        try:
            send(worker.stdin,audio=audio,rate=rate)
            response=receive(worker.stdout)
            if response is None:
                raise EOFError('Remote GPU inference exited')
            return response['boxes']
        except (BrokenPipeError,EOFError,OSError):
            stop_worker()
            if attempt==2:
                raise
            print('Reconnecting inference worker; completed recording cache retained',flush=True)
            time.sleep(5)
            worker=start_worker()
try:
    import birdsong_detect_distill.birdbox as frontend
    frontend.predict_boxes=predict
    from run_hexeberg_yolo import evaluate
    evaluate(args.job,args.dataset)
finally:
    stop_worker()
