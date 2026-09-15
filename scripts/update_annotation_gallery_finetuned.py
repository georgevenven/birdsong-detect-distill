#!/usr/bin/env python3
"""Update the existing 50 gallery examples with the fine-tuned Large 3,205s model."""
import copy
import hashlib
import json
from pathlib import Path

import make_annotation_gallery as original
from birdsong_detect_distill.full_encoder_linear import ARCHITECTURE, load_heads
from birdsong_detect_distill.model import load_backbone
from prepare_detector_study import verify

OUT = original.ROOT / 'results/annotation_gallery_finetuned_large_3205s_2026-09-14'
STUDY = original.ROOT / 'results/qwen_teacher_powdermill/full_encoder_backbones_snapshot_2026-09-14'
PREVIOUS = original.ROOT / 'results/annotation_gallery_linear_2k_2026-09-14'
ASSETS = original.PUBLIC / 'examples-finetuned-large-3205s'
np, torch = original.np, original.torch


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if any(original.digest(p) != sha for p,sha in plan['protected'].items()):
            raise ValueError('frozen gallery inputs changed')
        return plan
    study = json.loads((STUDY / 'manifest.json').read_text()); verify(study)
    report_path = STUDY / 'runs/large/comparison.json'
    report = json.loads(report_path.read_text()); protocol = report['protocol']; saved = protocol['checkpoint']
    checkpoint = Path(protocol['checkpoint_path'])
    if (saved['detector_architecture'] != ARCHITECTURE or saved['seed'] != 0
            or saved['metrics']['train_timebins'] != 641000 or saved['metrics']['selected_epoch'] != 3
            or original.digest(checkpoint) != protocol['checkpoint_sha256']):
        raise ValueError('unexpected fine-tuned Large checkpoint')
    gallery = json.loads((original.PUBLIC / 'gallery.json').read_text())
    if original.digest(original.PUBLIC/'gallery.json') != json.loads((PREVIOUS/'complete.json').read_text())['gallery_sha256']:
        raise ValueError('gallery differs from the previous published linear update')
    source = json.loads((original.OUT / 'manifest.json').read_text())
    entries = {e['id']:e for d in gallery['datasets'] for e in d['examples']}
    if len(entries) != 50 or set(entries) != {e['id'] for e in source['examples']}:
        raise ValueError('gallery selection changed')
    protected = {**study['protected_files'], **study['code_sha256']}
    paths = [Path(__file__), report_path, checkpoint, STUDY/'manifest.json', PREVIOUS/'complete.json',
        original.OUT/'manifest.json', original.ROOT/'scripts/make_annotation_gallery.py']
    for item in source['examples']:
        entry_path = original.OUT / f'{item["id"]}.json'
        entry = json.loads(entry_path.read_text())
        if any(entries[item['id']][key] != entry[key] for key in ['recording','start','duration','events','ground_truth']):
            raise ValueError('source excerpt or reference labels changed')
        arrays = original.OUT / 'arrays' / f'{item["id"]}.npz'
        if original.digest(arrays) != entry['arrays_sha256']:
            raise ValueError('original spectrogram cache changed')
        for name,sha in entry['image_sha256'].items():
            p = original.PUBLIC / 'examples' / name
            if original.digest(p) != sha: raise ValueError('original image changed')
            protected[str(p)] = sha
        paths += [entry_path, arrays, original.PUBLIC / entries[item['id']]['prediction'].lstrip('/')]
        if Path(item['record']['name']).stem in set(saved['training_recording_ids']) | set(saved['validation_recording_ids']):
            raise ValueError('gallery source overlaps detector fitting data')
    protected.update({str(p):original.digest(p) for p in paths})
    plan = dict(checkpoint=str(checkpoint),checkpoint_sha256=protocol['checkpoint_sha256'],checkpoint_metadata=saved,
        threshold=report['summary']['threshold'],threshold_source='Powdermill Recordings_2–4; unchanged calibration procedure',
        previous_gallery=gallery,previous_gallery_sha256=original.digest(original.PUBLIC/'gallery.json'),
        examples=source['examples'],inference=source['inference'],selection=source['selection'],protected=protected,
        reference_policy='Same 50 examples, human references, spectrogram scales, and full-source inference; no reselection.')
    original.write_json(path,plan)
    return plan


def main():
    OUT.mkdir(exist_ok=True); plan = prepare()
    if (OUT/'complete.json').exists():
        print('Already completed:', OUT); return
    if torch.cuda.mem_get_info(0)[0] < 20*1024**3:
        raise ValueError('GPU occupied; preserve other jobs')
    ASSETS.mkdir(exist_ok=True); (OUT/'arrays').mkdir(exist_ok=True)
    saved,device = plan['checkpoint_metadata'],torch.device('cuda:0')
    checkpoint = Path(plan['checkpoint'])
    backbone = load_backbone(saved['backbone_id'],device,revision=saved['backbone_revision'])
    head = load_heads([checkpoint],backbone,device)[checkpoint.stem]
    original.plt.style.use('default')
    original.plt.rcParams.update({'font.size':12,'axes.labelsize':11,'axes.edgecolor':'#8993a2',
        'text.color':'#17233b','axes.labelcolor':'#40516a','xtick.color':'#40516a','ytick.color':'#40516a'})
    gallery = copy.deepcopy(plan['previous_gallery'])
    gallery.update(checkpoint=checkpoint.name,checkpoint_sha256=plan['checkpoint_sha256'],threshold=plan['threshold'],
        training_seconds=3205,validation_seconds=800,seed=0,selected_epoch=3,head='Linear(768, 32) + sigmoid',
        head_parameters=24608,encoder_parameters=saved['audit']['encoder_parameters'],
        encoder_finetuned=True,architecture=ARCHITECTURE,updated='2026-09-14')
    entries = {e['id']:e for d in gallery['datasets'] for e in d['examples']}
    manifest_sha = original.digest(OUT/'manifest.json')
    for index,item in enumerate(plan['examples'],1):
        entry = json.loads((original.OUT/f'{item["id"]}.json').read_text())
        path = ASSETS/f'{item["id"]}-prediction.png'; record_path = OUT/f'{item["id"]}.json'
        arrays = OUT/'arrays'/f'{item["id"]}.npz'
        if record_path.exists():
            result = json.loads(record_path.read_text())
            if (result['manifest_sha256'] != manifest_sha or original.digest(path) != result['image_sha256']
                    or original.digest(arrays) != result['arrays_sha256']):
                raise ValueError('cached rendering changed')
        else:
            wave,sr = original.read_audio(item['record']); audio,_ = original.model_audio(wave,sr,'songmae')
            if hashlib.sha256(audio.tobytes()).hexdigest() != entry['audio_sha256']:
                raise ValueError('source waveform differs from original gallery')
            probability = original.smooth(original.songmae_probability(backbone,head,audio,device))
            left,right = round(entry['start']*200),round((entry['start']+entry['duration'])*200)
            crop = probability[:,left:right]; mask = crop.astype(np.float64) >= plan['threshold']
            if not np.isfinite(crop).all(): raise ValueError('nonfinite prediction')
            with np.load(original.OUT/'arrays'/f'{item["id"]}.npz') as cached:
                if mask.shape != cached['spectrogram'].shape: raise ValueError('spectrogram extent mismatch')
                original.panel(path,cached['spectrogram'],cached['events'],mask,cached['ignored'],entry['duration'],item['temporal_only'],True)
            np.savez_compressed(arrays,smoothed_probability=crop,mask=mask)
            result = dict(id=item['id'],recording=entry['recording'],start=entry['start'],duration=entry['duration'],
                audio_sha256=entry['audio_sha256'],manifest_sha256=manifest_sha,arrays_sha256=original.digest(arrays),
                image_sha256=original.digest(path),foreground_fraction=float(mask.mean()))
            original.write_json(record_path,result)
        entries[item['id']]['prediction'] = f'/{ASSETS.name}/{path.name}'
        original.write_json(OUT/'status.json',dict(state='rendering',completed=index,total=50))
        print(f'{index}/50 {item["id"]}: {entry["recording"]}',flush=True)
    if any(original.digest(p) != sha for p,sha in plan['protected'].items()):
        raise ValueError('frozen model or references changed')
    if original.digest(original.PUBLIC/'gallery.json') != plan['previous_gallery_sha256']:
        raise ValueError('gallery changed concurrently')
    original.write_json(OUT/'gallery.json',gallery); original.write_json(original.PUBLIC/'gallery.json',gallery)
    original.write_json(OUT/'complete.json',dict(state='complete',completed=50,original_images_unchanged=True,
        gallery_sha256=original.digest(original.PUBLIC/'gallery.json'),manifest_sha256=manifest_sha))
    original.write_json(OUT/'status.json',dict(state='complete',completed=50,total=50))


if __name__=='__main__':
    main()
