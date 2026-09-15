#!/usr/bin/env python3
"""Matched three-model XC comparisons from verified cached arrays; no inference."""
import json
from pathlib import Path

import plot_bare_linear_xc as base

np, plt = base.np, base.plt
OUT = base.ROOT / 'results/xc_three_model_comparison_2026-09-14'
SOURCES = [('bare_linear', 'Frozen encoder + linear'),
    ('transformer', 'Frozen encoder + transformer'),
    ('full_encoder', 'Fine-tuned encoder + linear')]
COLOR = '#00D5FF'


def prepare():
    sources, protected = [], {str(Path(__file__)):base.digest(__file__)}
    paths = [base.ROOT / 'scripts/plot_bare_linear_xc.py']
    for name, label in SOURCES:
        folder = base.ROOT / f'results/{name}_xc_heldout_2026-09-14'
        plan = json.loads((folder / 'manifest.json').read_text())
        entries = json.loads((folder / 'examples.json').read_text())['examples']
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['state'] != 'complete' or len(entries) != 8:
            raise ValueError('eight completed examples required for each model')
        for filename, sha in complete['files'].items():
            p = folder / 'share' / filename
            if base.digest(p) != sha:
                raise ValueError('source figures or provenance changed')
            protected[str(p)] = sha
        paths += [folder / 'manifest.json', folder / 'examples.json', folder / 'complete.json']
        for entry in entries:
            p = folder / 'arrays' / f"{entry['stem']}.npz"
            if base.digest(p) != entry['arrays_sha256']:
                raise ValueError('cached predictions changed')
            protected[str(p)] = entry['arrays_sha256']
        sources.append(dict(name=name, label=label, folder=str(folder), threshold=plan['threshold'],
            checkpoint=plan['checkpoint'], entries=entries))
    protected.update({str(p):base.digest(p) for p in paths})
    plan = dict(sources=sources, protected=protected,
        rendering='Replot original arrays and binary masks without modifying scores, thresholds, smoothing or contours.',
        layout='Each individual figure: three model columns, clean inputs on top and predictions below. Overview: eight matched excerpt rows and three prediction columns.',
        reference='Clean inputs are not human ground-truth annotations. Cyan contours are model predictions.',
        postprocessing='Same full-source probability smoothing sigma (2 mel,3 frames); original per-model Powdermill Recordings_2–4 thresholds retained.')
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('frozen comparison inputs changed')
    base.write_json(path, plan)
    return plan


def cached(source, index, reference):
    entry = source['entries'][index]
    for key in ['stem','recording','excerpt_frame','source_start','source_end','full_source_spectrogram_sha256']:
        if entry[key] != reference[key]:
            raise ValueError('examples are not temporally/source matched')
    path = Path(source['folder']) / 'arrays' / f"{entry['stem']}.npz"
    with np.load(path) as value:
        display, mask = value['spectrogram_db'].copy(), value['mask'].copy()
        if not np.array_equal(mask, value['smoothed_probability'].astype(np.float64) >= source['threshold']):
            raise ValueError('cached mask does not match its original threshold')
    if display.shape != (128,1000) or mask.shape != display.shape:
        raise ValueError('unexpected excerpt geometry')
    return display, mask


def panel(ax, display, mask=None):
    ax.imshow(display, origin='lower', aspect='auto', extent=(0,5,0,128), cmap='magma',
        vmin=-70, vmax=0, interpolation='nearest', rasterized=True)
    if mask is not None and mask.any():
        ax.contour((np.arange(1002)-.5)/200, np.arange(130)-.5, np.pad(mask,1),
            levels=[.5], colors=COLOR, linewidths=.65)
    ticks = (base.librosa.hz_to_mel([1000,2000,4000,8000,16000])-base.librosa.hz_to_mel(20))*128/(
        base.librosa.hz_to_mel(16000)-base.librosa.hz_to_mel(20))
    ax.set(xlim=(0,5), ylim=(0,128), xticks=range(6))
    ax.set_yticks(ticks, labels=['1','2','4','8','16'])


def save(fig, share, stem, dpi):
    for ext in ['png','pdf']:
        fig.savefig(share / f'{stem}.{ext}', dpi=dpi, facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = prepare()
    if (OUT / 'complete.json').exists():
        print('Already rendered:', OUT)
        return
    share = OUT / 'share'
    share.mkdir(exist_ok=True)
    plt.style.use('default')
    plt.rcParams.update({'font.size':11, 'axes.titlesize':11, 'axes.labelsize':11,
        'xtick.labelsize':10, 'ytick.labelsize':10, 'pdf.fonttype':42})
    overview, grid = plt.subplots(8,3, figsize=(11.5,17.5), sharex=True, sharey=True)
    overview.subplots_adjust(left=.07, right=.99, bottom=.04, top=.915, hspace=.62, wspace=.13)
    overview.suptitle('Same held-out XC excerpts · three detector variants\n'
        '2,000 s training · seed 0 · unchanged post-processing · cyan = predicted foreground', fontsize=15, y=.99)
    for col, source in enumerate(plan['sources']):
        box = grid[0,col].get_position()
        overview.text((box.x0+box.x1)/2, .938, f"{source['label']}\nThreshold {source['threshold']:.2f}",
            ha='center', va='center', fontsize=11)
    for index, reference in enumerate(plan['sources'][0]['entries']):
        start = reference['excerpt_frame']/200
        title = f"{reference['recording']} · {start:g}–{start+5:g} s"
        fig, axes = plt.subplots(2,3, figsize=(11.5,4.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.99, bottom=.15, top=.77, hspace=.68, wspace=.13)
        fig.suptitle(title + ' · 2,000 s training', fontsize=15, y=.98)
        first = None
        for col, source in enumerate(plan['sources']):
            display, mask = cached(source, index, reference)
            if first is not None and not np.array_equal(display, first):
                raise ValueError('spectrogram pixels differ between models')
            first = display
            panel(axes[0,col], display)
            panel(axes[1,col], display, mask)
            axes[0,col].set_title(source['label'] + '\n(a) Clean input', pad=8)
            axes[1,col].set_title(f"(b) Prediction · threshold {source['threshold']:.2f}", pad=7)
            panel(grid[index,col], display, mask)
            grid[index,col].tick_params(labelbottom=True)
        grid[index,0].set_title(title, loc='left', fontsize=11, pad=6)
        fig.supylabel('Frequency (kHz; mel-spaced)', x=.01, fontsize=11)
        fig.supxlabel('Time within excerpt (s)', y=.035, fontsize=11)
        save(fig, share, reference['stem'], 400)
        print('Matched comparison:', reference['stem'], flush=True)
    overview.supylabel('Frequency (kHz; mel-spaced)', x=.008, fontsize=12)
    overview.supxlabel('Time within excerpt (s)', y=.012, fontsize=12)
    save(overview, share, '00_OVERVIEW', 250)
    attribution = (Path(plan['sources'][0]['folder']) / 'share/README.md').read_text().split('## Recording attribution',1)[1]
    notes = ['# Three-model held-out XC comparisons', '',
        'Open 00_OVERVIEW.png first: eight excerpt rows, three model columns. Individual PNG/PDFs retain clean input above each prediction.',
        'Left: frozen SongMAE-Large + linear (threshold 0.07). Middle: frozen SongMAE-Large + one 128-wide transformer layer (0.05). Right: fully fine-tuned SongMAE-Large encoder + linear (0.09).',
        'All models: seed 0, the same 2,000 s XC training / 800 s validation split, checkpoint selection by XC validation BCE.',
        'Exact original source recordings, excerpt times, spectrograms and cached post-processed masks. No inference or training rerun. Model-specific thresholds were selected on Powdermill Recordings 2–4, not these examples.',
        plan['postprocessing'], plan['reference'],
        'Standard Matplotlib rendering. Cyan outlines follow the existing binary masks; no additional smoothing, morphology, contour cleanup or normalization changes.', '',
        '## Recording attribution' + attribution]
    (share / 'README.md').write_text('\n'.join(notes)+'\n')
    if any(base.digest(p) != sha for p,sha in plan['protected'].items()):
        raise ValueError('source artifacts changed while rendering')
    base.write_json(share / 'provenance.json', dict(manifest_sha256=base.digest(OUT / 'manifest.json'),
        sources=plan['sources'], source_sha256=plan['protected'], no_new_inference=True))
    hashes = {p.name:base.digest(p) for p in sorted(share.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'}
    (share / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
    base.write_json(OUT / 'complete.json', dict(state='complete', examples=8, files=hashes))
    print('Complete:', share, flush=True)


if __name__ == '__main__':
    main()
