#!/usr/bin/env python3
"""Wait for a helper's exact configured model before starting its annotation client."""
import json
import sys
import time
from pathlib import Path

import requests


def main():
    backend = json.loads(Path(sys.argv[1]).read_text())
    url = backend['url']
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            healthy = requests.get(url + '/health', timeout=5).ok
        except requests.RequestException:
            healthy = False
        if healthy:
            response = requests.get(url + '/props', timeout=10)
            response.raise_for_status()
            props = response.json()
            assert props['model_path'] == backend['remote_model_path']
            assert props['total_slots'] == backend['workers']
            assert props['default_generation_settings']['n_ctx'] == backend['context_per_slot']
            print('Helper ready: matching model, slots, and context.', flush=True)
            return
        time.sleep(2)
    raise TimeoutError('Helper did not become healthy within 180 seconds')


if __name__ == '__main__':
    main()
