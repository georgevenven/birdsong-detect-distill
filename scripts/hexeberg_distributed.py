"""Author-recipe training: one V100 per Nano seed, two Twins GPUs for Large."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from ultralytics import __version__
from ultralytics.models.yolo.detect import DetectionTrainer


class MigratedTrainer(DetectionTrainer):
    def check_resume(self, overrides):
        super().check_resume(overrides)
        # Relocate outputs/data without changing resumed optimization settings.
        for key in ('data', 'project', 'name', 'save_dir', 'workers', 'exist_ok'):
            if key in overrides:
                setattr(self.args, key, overrides[key])
        # get_cfg strips save_dir from overrides before this hook.
        self.args.save_dir = str(Path(self.args.project)/self.args.name)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if int(os.getenv('WORLD_SIZE', '1')) > 1 else model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', choices=['n', 'l'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    assert __version__ == '8.3.109'
    root = args.root
    job = f'yolo_{args.variant}_25000_s{args.seed}'
    out = root/'artifacts'/job
    done = out/'training_complete.json'
    if done.exists():
        return
    if (out/'delegated.json').exists():
        if int(os.getenv('RANK', '-1')) in {-1, 0}:
            subprocess.run([sys.executable, str(root/'scripts/await_hexeberg_remote.py'),
                            '--job', job], check=True)
        return
    plan = json.loads((root/'results/manifest.json').read_text())
    config = dict(plan['authors'][args.variant])
    last = out/'fit/weights/last.pt'
    config.update(model=str(root/f'models/coco_yolo11{args.variant}.pt'),
        data=str(root/'dataset/dataset.yaml'), seed=args.seed,
        device='0,1' if args.variant == 'l' else os.environ.get('CUDA_VISIBLE_DEVICES','0'),
        workers=int(os.getenv('YOLO_WORKERS', '4' if args.variant=='l' else '1')),
        project=str(out), name='fit', save_dir=str(out/'fit'), exist_ok=True, plots=False)
    if last.exists():
        config.update(model=str(last), resume=str(last))
    try:
        trainer = MigratedTrainer(overrides=config)
        trainer.train()
    except BaseException as error:
        if int(os.getenv('RANK', '-1')) in {-1, 0}:
            out.mkdir(parents=True, exist_ok=True)
            (out/'training_failed.json').write_text(json.dumps(dict(error=repr(error), time=time.time()))+'\n')
        raise
    if int(os.getenv('RANK', '-1')) in {-1, 0}:
        shutil.copy2(out/'fit/weights/best.pt', out/'model.pt')
        temporary = done.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(job=job, finished=time.time(),
            torch=torch.__version__, ultralytics=__version__,
            hardware=f'{os.uname().nodename} DDP SyncBatchNorm' if args.variant=='l' else 'V100 FP16',
            resumed=bool(config.get('resume'))), indent=2)+'\n')
        temporary.replace(done)


if __name__ == '__main__':
    main()
