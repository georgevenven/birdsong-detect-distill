#!/usr/bin/env python3
"""The trained 2k transformer head on the frozen bare-linear XC examples."""
import hashlib
import json
from pathlib import Path

import plot_bare_linear_xc as base
from birdsong_detect_distill.model import load_detector

np, plt, torch = base.np, base.plt, base.torch
OUT = base.ROOT / 'results/transformer_xc_heldout_2026-09-14'
STUDY = base.ROOT / 'results/qwen_teacher_powdermill/new_teacher_2000train_800val_2026-09-14'
REPORT = STUDY / 'runs/layers1_d128_new2k_s0/comparison.json'


def prepare():
    original = base.prepare()
    report = json.loads(REPORT.read_text())
    saved = report['protocol']['checkpoint']
    linear = json.loads((base.STUDY / 'runs/bare_linear_d0_new2k_s0/comparison.json').read_text())['protocol']['checkpoint']
    for key in ['seed', 'training_recording_ids', 'validation_recording_ids', 'backbone_id',
                'backbone_revision', 'annotations_sha256', 'validation_annotations_sha256']:
        if saved[key] != linear[key]:
            raise ValueError(f'comparison differs in {key}')
    if saved['head_layers'] != 1 or saved['hidden'] != 128 or saved['metrics']['train_timebins'] != 400000:
        raise ValueError('unexpected transformer checkpoint')
    if base.digest(saved['checkpoint']) != saved['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    study = json.loads((STUDY / 'manifest.json').read_text())
    base.verify(study)
    plan = {**original, 'checkpoint': saved['checkpoint'], 'threshold': report['summary']['threshold'],
        'architecture': 'Frozen SongMAE-Large; 128-wide, one-layer self-attention detector',
        'trainable_parameters': saved['trainable_parameters'], 'checkpoint_metadata': saved,
        'calibration_report': str(REPORT)}
    plan['protected'] = {**original['protected'], **study['code_sha256']}
    for path in [Path(__file__), REPORT, STUDY / 'manifest.json', Path(saved['checkpoint']),
                 base.OUT / 'manifest.json', base.OUT / 'examples.json', base.OUT / 'share/README.md']:
        plan['protected'][str(path)] = base.digest(path)
    path = OUT / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('frozen transformer figure dependencies changed')
    else:
        base.write_json(path, plan)
    return plan


def panels(axes, entry, display, mask, overview=False):
    base.panels(axes, entry, display, mask, overview)
    axes[1].set_title('(b) Transformer, post-processed', pad=6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = prepare()
    if (OUT / 'complete.json').exists():
        print('Already rendered:', OUT)
        return
    if torch.cuda.mem_get_info(0)[0] < 20 * 1024**3:
        raise ValueError('GPU occupied; preserve other jobs')
    device = torch.device('cuda:0')
    backbone, head, _ = load_detector(plan['checkpoint'], device, plan['revision'])
    if sum(p.numel() for p in head.parameters()) != plan['trainable_parameters']:
        raise ValueError('unexpected head size')
    share, cache = OUT / 'share', OUT / 'arrays'
    share.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    originals = json.loads((base.OUT / 'examples.json').read_text())['examples']
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42})
    overview, grid = plt.subplots(4, 4, figsize=(14, 9.5))
    overview.subplots_adjust(left=.045, right=.99, bottom=.055, top=.87, hspace=.85, wspace=.28)
    overview.suptitle('Held-out XC · SongMAE-Large + transformer head · 2,000 s training\n'
        f"Probability smoothing + Powdermill threshold {plan['threshold']:.2f}; cyan = predicted foreground",
        fontsize=14, y=.99)
    entries = []
    for i, (entry, previous) in enumerate(zip(plan['examples'], originals, strict=True)):
        raw = base.load_spec_slice(base.XC / 'shards' / entry['shard'], entry['source_start'], entry['source_end'])
        source_hash = hashlib.sha256(raw.tobytes()).hexdigest()
        if source_hash != previous['full_source_spectrogram_sha256']:
            raise ValueError('input differs from the linear figures')
        probability = base.predict(raw, backbone, head, device)
        smoothed = base.smooth(probability)
        left, right = entry['excerpt_frame'], entry['excerpt_frame'] + 1000
        display = (raw - float(raw.max()))[:, left:right]
        mask = (smoothed.astype(np.float64) >= plan['threshold'])[:, left:right]
        name = previous['stem']
        original_cache = base.OUT / 'arrays' / f'{name}.npz'
        if base.digest(original_cache) != previous['arrays_sha256']:
            raise ValueError('original cached input changed')
        with np.load(original_cache) as original:
            if not np.array_equal(display, original['spectrogram_db']):
                raise ValueError('displayed excerpt differs')
        np.savez_compressed(cache / f'{name}.npz', spectrogram_db=display,
            raw_probability=probability[:, left:right], smoothed_probability=smoothed[:, left:right], mask=mask)
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=.17, right=.98, bottom=.14, top=.83, hspace=.55)
        fig.suptitle(f"{entry['recording']} · {left / 200:g}–{right / 200:g} s", fontsize=11, y=.98)
        panels(axes, entry, display, mask)
        fig.text(.025, .49, 'Frequency (kHz; mel-spaced)', rotation=90, va='center', fontsize=9)
        for ext in ['png', 'pdf']:
            fig.savefig(share / f'{name}.{ext}', dpi=600, facecolor='white')
        plt.close(fig)
        row, col = (i // 4) * 2, i % 4
        panels([grid[row, col], grid[row + 1, col]], entry, display, mask, True)
        entries.append(dict(**entry, stem=name, full_source_spectrogram_sha256=source_hash,
            full_probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
            foreground_fraction=float(mask.mean()), arrays_sha256=base.digest(cache / f'{name}.npz')))
        print(name, 'foreground fraction', round(float(mask.mean()), 4), flush=True)
    for ext in ['png', 'pdf']:
        overview.savefig(share / f'00_OVERVIEW.{ext}', dpi=250, facecolor='white')
    plt.close(overview)
    base.write_json(OUT / 'examples.json', dict(manifest_sha256=base.digest(OUT / 'manifest.json'), examples=entries))
    attribution = (base.OUT / 'share/README.md').read_text().split('## Recording attribution', 1)[1]
    notes = ['# Held-out XC: transformer SongMAE detector', '',
        'Open 00_OVERVIEW.png first. Eight square 3.5-inch PNG/PDF pairs; PNGs are 600 dpi.',
        'Top: clean spectrogram, NOT human ground truth. Bottom: cyan contours of actual predictions.',
        'Frozen SongMAE-Large + one 128-wide self-attention layer, 365,088 head parameters. Seed 0; 2,000 s training; 800 s separate XC validation; best validation-BCE checkpoint (epoch 4). No new training.',
        f"Probability Gaussian smoothing: sigma 2 mel bins and 3 frames (15 ms). Threshold {plan['threshold']:.2f} selected on Powdermill Recordings 2–4, not these XC examples. No other cleanup.",
        'Identical recordings, excerpts, inputs, normalization, full-source inference/overlap, smoothing and display as the bare-linear figures. The linear threshold was 0.07; each model uses its own pre-calibrated threshold.',
        plan['inference'] + '.', plan['selection'] + '.', plan['exclusions'] + '.',
        'Qualitative examples only; no human localization labels or benchmark metrics.', '',
        '## Recording attribution' + attribution]
    (share / 'README.md').write_text('\n'.join(notes) + '\n')
    base.write_json(share / 'provenance.json', dict(checkpoint=plan['checkpoint_metadata'],
        threshold=plan['threshold'], manifest_sha256=base.digest(OUT / 'manifest.json'), examples=entries))
    hashes = {p.name: base.digest(p) for p in sorted(share.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'}
    (share / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name, sha in hashes.items()))
    base.write_json(OUT / 'complete.json', dict(state='complete', examples=len(entries), files=hashes))
    print('Complete:', share, flush=True)


if __name__ == '__main__':
    main()
