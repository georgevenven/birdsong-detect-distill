"""Verify every transferred training image and label against original receipts."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])/'dataset'
receipts = list((root/'provenance').glob('*.json'))
assert len(receipts) == 5500, len(receipts)
for path in receipts:
    record = json.loads(path.read_text())
    for folder, suffix, key in [('images', '.png', 'image_sha256'), ('labels', '.txt', 'labels_sha256')]:
        asset = root/folder/record['partition']/(record['recording']+suffix)
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == record[key], str(asset)
print('Verified all 5500 images and labels', flush=True)
