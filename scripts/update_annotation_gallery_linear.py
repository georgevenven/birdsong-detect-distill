#!/usr/bin/env python3
"""Replace gallery predictions, preserving all 50 source-frozen examples and references."""
import copy
import hashlib
import json
from pathlib import Path

import make_annotation_gallery as old
from birdsong_detect_distill.model import load_backbone
from birdsong_detect_distill.pointwise_bare_linear import ARCHITECTURE, load_heads

np, torch = old.np, old.torch
OUT = old.ROOT / 'results/annotation_gallery_linear_2k_2026-09-14'
REPORT = old.ROOT / 'results/qwen_teacher_powdermill/pointwise_bare_linear_2000train_800val_2026-09-14/runs/bare_linear_d0_new2k_s0/comparison.json'
ASSETS = old.PUBLIC / 'examples-linear-2k'


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if any(old.digest(p) != sha for p, sha in plan['protected'].items()):
            raise ValueError('frozen gallery inputs changed')
        return plan
    report = json.loads(REPORT.read_text())
    saved = report['protocol']['checkpoint']
    if saved['seed'] != 0 or saved['detector_architecture'] != ARCHITECTURE or saved['metrics']['train_timebins'] != 400000:
        raise ValueError('unexpected linear detector')
    if old.digest(saved['checkpoint']) != saved['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    previous = json.loads((old.OUT / 'manifest.json').read_text())
    gallery = json.loads((old.PUBLIC / 'gallery.json').read_text())
    if gallery['checkpoint_sha256'] != previous['checkpoint_sha256']:
        raise ValueError('live gallery does not match the original 20k gallery')
    entries = {e['id']:e for d in gallery['datasets'] for e in d['examples']}
    if len(entries) != 50 or set(entries) != {e['id'] for e in previous['examples']}:
        raise ValueError('gallery selection changed')
    protected = {str(p):old.digest(p) for p in [Path(__file__), REPORT, Path(saved['checkpoint']),
        old.OUT / 'manifest.json', old.ROOT / 'scripts/make_annotation_gallery.py',
        old.ROOT / 'scripts/evaluate_2d.py', old.ROOT / 'scripts/evaluate_songmae_smoothing.py',
        old.ROOT / 'src/birdsong_detect_distill/pointwise_bare_linear.py',
        old.ROOT / 'src/birdsong_detect_distill/model.py', old.ROOT / 'src/birdsong_detect_distill/benchmark_data.py']}
    for item in previous['examples']:
        entry_path = old.OUT / f"{item['id']}.json"
        entry = json.loads(entry_path.read_text())
        for key in ['recording', 'start', 'duration', 'events', 'ground_truth', 'prediction']:
            if entries[item['id']][key] != entry[key]:
                raise ValueError('example differs from frozen selection')
        for name, sha in entry['image_sha256'].items():
            p = old.PUBLIC / 'examples' / name
            if old.digest(p) != sha:
                raise ValueError('original gallery image changed')
            protected[str(p)] = sha
        p = old.OUT / 'arrays' / f"{item['id']}.npz"
        if old.digest(p) != entry['arrays_sha256']:
            raise ValueError('original gallery arrays changed')
        protected[str(p)], protected[str(entry_path)] = entry['arrays_sha256'], old.digest(entry_path)
        if Path(item['record']['name']).stem in set(saved['training_recording_ids']) | set(saved['validation_recording_ids']):
            raise ValueError('gallery recording occurs in detector fitting data')
    plan = dict(checkpoint=saved, threshold=report['summary']['threshold'], protected=protected,
        previous_gallery=gallery, previous_gallery_sha256=old.digest(old.PUBLIC / 'gallery.json'),
        examples=previous['examples'], inference=previous['inference'],
        reference_policy='Original images, events, audio excerpts and display scales unchanged; no example reselection.',
        selection=previous['selection'])
    old.write_json(path, plan)
    return plan


def main():
    OUT.mkdir(exist_ok=True)
    plan = prepare()
    if (OUT / 'complete.json').exists():
        print('Already completed:', OUT)
        return
    if torch.cuda.mem_get_info(0)[0] < 20 * 1024**3:
        raise ValueError('GPU occupied; preserve existing work')
    (OUT / 'arrays').mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    saved, device = plan['checkpoint'], torch.device('cuda:0')
    checkpoint = Path(saved['checkpoint'])
    backbone = load_backbone(saved['backbone_id'], device, revision=saved['backbone_revision'])
    head = load_heads([checkpoint], backbone, device)[checkpoint.stem]
    old.plt.style.use('default')
    old.plt.rcParams.update({'font.size':12,'axes.labelsize':11,'axes.edgecolor':'#8993a2',
        'text.color':'#17233b','axes.labelcolor':'#40516a','xtick.color':'#40516a','ytick.color':'#40516a'})
    gallery = copy.deepcopy(plan['previous_gallery'])
    gallery.update(checkpoint=checkpoint.name, checkpoint_sha256=saved['checkpoint_sha256'],
        threshold=plan['threshold'], training_seconds=2000, seed=0, head='Linear(768, 32) + sigmoid',
        head_parameters=24608, updated='2026-09-14')
    entries = {e['id']:e for d in gallery['datasets'] for e in d['examples']}
    for index, item in enumerate(plan['examples'], 1):
        entry = json.loads((old.OUT / f"{item['id']}.json").read_text())
        path = ASSETS / f"{item['id']}-prediction.png"
        record_path = OUT / f"{item['id']}.json"
        if record_path.exists():
            result = json.loads(record_path.read_text())
            if result['manifest_sha256'] != old.digest(OUT / 'manifest.json') or old.digest(path) != result['image_sha256']:
                raise ValueError('cached linear rendering changed')
        else:
            wave, sr = old.read_audio(item['record'])
            audio, sr = old.model_audio(wave, sr, 'songmae')
            if hashlib.sha256(audio.tobytes()).hexdigest() != entry['audio_sha256']:
                raise ValueError('input waveform differs from the original gallery')
            probability = old.smooth(old.songmae_probability(backbone, head, audio, device))
            left, right = round(entry['start'] * 200), round((entry['start'] + entry['duration']) * 200)
            crop = probability[:, left:right]
            if not np.isfinite(crop).all():
                raise ValueError('nonfinite prediction')
            mask = crop.astype(np.float64) >= plan['threshold']
            with np.load(old.OUT / 'arrays' / f"{item['id']}.npz") as cached:
                spec, events, ignored = cached['spectrogram'], cached['events'], cached['ignored']
                if mask.shape != spec.shape:
                    raise ValueError('prediction and original spectrogram extents differ')
                old.panel(path, spec, events, mask, ignored, entry['duration'], item['temporal_only'], True)
            arrays = OUT / 'arrays' / f"{item['id']}.npz"
            np.savez_compressed(arrays, smoothed_probability=crop, mask=mask)
            result = dict(id=item['id'], recording=entry['recording'], start=entry['start'], duration=entry['duration'],
                audio_sha256=entry['audio_sha256'], manifest_sha256=old.digest(OUT / 'manifest.json'),
                arrays_sha256=old.digest(arrays), image_sha256=old.digest(path), foreground_fraction=float(mask.mean()))
            old.write_json(record_path, result)
        entries[item['id']]['prediction'] = f'/examples-linear-2k/{path.name}'
        old.write_json(OUT / 'status.json', dict(state='rendering', completed=index, total=50))
        print(f"{index}/50 {item['id']}: {entry['recording']} at {entry['start']:.2f} s", flush=True)
    if any(old.digest(p) != sha for p, sha in plan['protected'].items()):
        raise ValueError('original gallery or model changed during rendering')
    if old.digest(old.PUBLIC / 'gallery.json') != plan['previous_gallery_sha256']:
        raise ValueError('gallery metadata changed concurrently')
    old.write_json(OUT / 'gallery.json', gallery)
    old.write_json(old.PUBLIC / 'gallery.json', gallery)
    old.write_json(OUT / 'complete.json', dict(state='complete', completed=50,
        gallery_sha256=old.digest(old.PUBLIC / 'gallery.json'), original_images_unchanged=True,
        manifest_sha256=old.digest(OUT / 'manifest.json')))
    print('Complete: 50 new linear predictions; all references and old prediction images preserved.', flush=True)


if __name__ == '__main__':
    main()
