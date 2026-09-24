"""One complete Large evaluation per Twins GPU, using the canonical scorer."""
import argparse
import fcntl
import time
import torch
import hexeberg_suite as s
from queue_hexeberg_migration import AUDIT

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--gpu',type=int,required=True)
args=parser.parse_args()
job=f'yolo_l_25000_s{args.seed}'
out=s.ART/job
report=s.OUT/'runs'/job
def status(state,**extra):
    s.atomic(report/'local_evaluation_status.json',dict(state=state,gpu=args.gpu,updated=time.time(),**extra))
try:
    s.prepare()
    s.frozen(report/'checkpoint.json',dict(path=str(out/'model.pt'),sha256=s.sha(out/'model.pt'),seed=args.seed,
        selection='Ultralytics native validation fitness, best.pt',manifest_sha256=s.sha(s.OUT/'manifest.json'),
        train_args_sha256=s.sha(out/'fit/args.yaml'),history_sha256=s.sha(out/'fit/results.csv'),
        initialization_sha256=s.sha(s.ROOT/'models/coco_yolo11l.pt'),
        migration_manifest_sha256=s.sha(AUDIT/'manifest.json'),hardware=s.read(out/'training_complete.json'),
        evaluation_execution=dict(host='Lambda-Twins',gpu=args.gpu,torch=torch.__version__,
            adapter_sha256=s.sha(__file__),frontend_sha256=s.sha(s.ROOT/'src/birdsong_detect_distill/birdbox.py'))))
    import birdsong_detect_distill.birdbox as frontend
    original=frontend.predict_boxes
    def predict(model,audio,rate,device,**kwargs):
        return original(model,audio,rate,f'cuda:{args.gpu}',**kwargs)
    frontend.predict_boxes=predict
    from run_hexeberg_yolo import evaluate
    folder=s.OUT/'external'/job
    folder.mkdir(parents=True,exist_ok=True)
    for dataset in ['powdermill','wabad','hawaii','xcsl','nips4bplus']:
        status('evaluating',dataset=dataset)
        with (folder/f'{dataset}.dispatch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            evaluate(job,dataset)
    status('complete')
    s.atomic(report/'status.json',dict(state='complete',updated_unix=time.time()))
except BaseException as error:
    status('failed',error=repr(error))
    raise
