"""Retrieve a delegated checkpoint before the original queue evaluates it."""
import argparse
import subprocess
import time
import hexeberg_suite as s
from queue_hexeberg_migration import SSH, REMOTE, transfer

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--job', required=True)
args=parser.parse_args()
if (s.ART/args.job/'resumed_on_twins.json').exists():
    while not (s.ART/args.job/'training_complete.json').exists():
        if (s.ART/args.job/'training_failed.json').exists():
            raise RuntimeError('Twins resumed training failed: '+args.job)
        time.sleep(60)
    transfer(str(s.ART/args.job)+'/',f'{SSH[-1]}:{REMOTE}/artifacts/{args.job}/')
    raise SystemExit(0)
while True:
    command=(f'if test -f {REMOTE}/artifacts/{args.job}/training_complete.json; then echo complete; '
             f'elif test -f {REMOTE}/artifacts/{args.job}/training_failed.json; then echo failed; '
             'else echo waiting; fi')
    try:
        result=subprocess.run(SSH+[command],check=True,capture_output=True,text=True,timeout=45).stdout.strip()
    except (subprocess.SubprocessError, OSError) as error:
        print('Remote check unavailable; retrying:',error,flush=True)
        time.sleep(60)
        continue
    if result=='failed':
        raise RuntimeError('Delegated training failed: '+args.job)
    if result=='complete':
        transfer(f'{SSH[-1]}:{REMOTE}/artifacts/{args.job}/',str(s.ART/args.job)+'/')
        break
    time.sleep(60)
