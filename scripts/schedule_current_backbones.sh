#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
annotation=birdsong-qwen-xc-continuation.service
server=songmae-qwen.service
mode=${1:---after-qwen}
runner=${2:-scripts/run_current_backbones.sh}
case "$runner" in
  scripts/run_current_backbones.sh|scripts/run_current_scaling.sh|scripts/run_three_seed_figures.sh|scripts/run_training_duration.sh) ;;
  *) echo "Unknown figure runner: $runner" >&2; exit 1 ;;
esac
resume_annotation=0
restore_server=0

restore() {
  if [[ -f /home/george-vengrovski/.config/birdsong-detect-distill/qwen.paused ]]; then
    echo "Qwen remains manually paused; explicit user restart required."
    return
  fi
  if (( restore_server )); then systemctl --user start "$server"; fi
  if (( resume_annotation )); then systemctl --user start --no-block "$annotation"; fi
}
trap restore EXIT
trap 'exit 130' INT TERM
case "$mode" in
  --pause-qwen)
    if systemctl --user is-active --quiet "$annotation"; then
      resume_annotation=1
      echo "Gracefully draining Qwen; it will resume after the figure run."
      systemctl --user stop "$annotation"
    fi ;;
  --after-qwen)
    echo "Queued behind XC Qwen annotations; no GPU work until that job exits."
    while true; do
      state=$(systemctl --user show "$annotation" -p ActiveState --value)
      case "$state" in
        active|activating|deactivating|reloading) sleep 30 ;;
        inactive|failed) break ;;
        *) echo "Unexpected Qwen state: $state" >&2; exit 1 ;;
      esac
    done ;;
  *) echo "Use --after-qwen or --pause-qwen" >&2; exit 1 ;;
esac

.venv/bin/python - <<'PY'
import json, shutil, subprocess
from urllib.request import urlopen
server_pid = subprocess.check_output(['systemctl','--user','show','songmae-qwen.service','-p','MainPID','--value'], text=True).strip()
pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).split()
if set(pids) - {server_pid}:
    raise SystemExit('Another GPU job is active; refusing to interfere.')
if server_pid != '0':
    with urlopen('http://127.0.0.1:8080/slots', timeout=10) as response:
        slots=json.load(response)
    if any(slot['is_processing'] for slot in slots):
        raise SystemExit('Qwen has active requests from another client; refusing to interrupt.')
if shutil.disk_usage('.').free < 6*1024**3:
    raise SystemExit('Need at least 6 GiB free for the new prediction caches.')
print('GPU ownership and disk checks passed.', flush=True)
PY
if systemctl --user is-active --quiet "$server"; then
  restore_server=1
  systemctl --user stop "$server"
fi
bash "$runner"
