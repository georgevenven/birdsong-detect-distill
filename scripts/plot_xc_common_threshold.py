#!/usr/bin/env python3
"""Re-threshold the existing three-model XC comparison at one shared value."""
import json
from pathlib import Path

import plot_xc_detector_comparison as original

base, np, plt = original.base, original.np, original.plt
THRESHOLD = .09
OUT = base.ROOT / 'results/xc_three_model_threshold009_2026-09-14'


def prepare():
    source = original.OUT / 'manifest.json'
    plan = json.loads(source.read_text())
    if any(base.digest(p) != sha for p,sha in plan['protected'].items()):
        raise ValueError('original comparison inputs changed')
    plan['protected'].update({str(p):base.digest(p) for p in [source, Path(__file__)]})
    for model in plan['sources']:
        model['original_threshold'] = model['threshold']
        model['threshold'] = THRESHOLD
    plan.update(common_threshold=THRESHOLD,
        rendering='Same cached continuous probabilities, smoothing, inputs and excerpt selection; only re-threshold all models at 0.09.',
        postprocessing='Original full-source probability Gaussian sigma (2 mel,3 frames) retained; float64 comparison against common threshold 0.09. No additional smoothing or morphology.',
        caveat='A shared numeric threshold does not guarantee matched precision, recall or score calibration. Qualitative operating-point check only; original calibrated evaluations remain unchanged.')
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('frozen common-threshold comparison changed')
    base.write_json(path, plan)
    return plan


def cached(model, index, reference):
    display, old_mask = original.cached({**model, 'threshold':model['original_threshold']}, index, reference)
    path = Path(model['folder']) / 'arrays' / f"{reference['stem']}.npz"
    with np.load(path) as values:
        mask = values['smoothed_probability'].astype(np.float64) >= THRESHOLD
    if THRESHOLD == model['original_threshold'] and not np.array_equal(mask, old_mask):
        raise ValueError('unchanged threshold changed the mask')
    if THRESHOLD >= model['original_threshold'] and np.any(mask & ~old_mask):
        raise ValueError('raising the threshold unexpectedly added foreground')
    return display, mask, dict(example=reference['stem'], model=model['name'],
        original_threshold=model['original_threshold'], common_threshold=THRESHOLD,
        original_foreground_pixels=int(old_mask.sum()), common_foreground_pixels=int(mask.sum()),
        removed_pixels=int((old_mask & ~mask).sum()), added_pixels=int((mask & ~old_mask).sum()))


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
    overview.suptitle('Same held-out XC excerpts · common threshold 0.09\n'
        '2,000 s training · seed 0 · same probabilities and smoothing · cyan = predicted foreground', fontsize=15, y=.99)
    for col, model in enumerate(plan['sources']):
        box = grid[0,col].get_position()
        overview.text((box.x0+box.x1)/2, .938, f"{model['label']}\nThreshold {THRESHOLD:.2f}",
            ha='center', va='center', fontsize=11)
    effects = []
    for index, reference in enumerate(plan['sources'][0]['entries']):
        start = reference['excerpt_frame']/200
        title = f"{reference['recording']} · {start:g}–{start+5:g} s"
        fig, axes = plt.subplots(2,3, figsize=(11.5,4.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.99, bottom=.15, top=.77, hspace=.68, wspace=.13)
        fig.suptitle(title + ' · common threshold 0.09', fontsize=15, y=.98)
        first = None
        for col, model in enumerate(plan['sources']):
            display, mask, effect = cached(model, index, reference)
            if first is not None and not np.array_equal(display, first):
                raise ValueError('spectrogram pixels differ between models')
            first = display
            effects.append(effect)
            original.panel(axes[0,col], display)
            original.panel(axes[1,col], display, mask)
            axes[0,col].set_title(model['label'] + '\n(a) Clean input', pad=8)
            axes[1,col].set_title('(b) Prediction · threshold 0.09', pad=7)
            original.panel(grid[index,col], display, mask)
            grid[index,col].tick_params(labelbottom=True)
        grid[index,0].set_title(title, loc='left', fontsize=11, pad=6)
        fig.supylabel('Frequency (kHz; mel-spaced)', x=.01, fontsize=11)
        fig.supxlabel('Time within excerpt (s)', y=.035, fontsize=11)
        original.save(fig, share, reference['stem'], 400)
        print('Common threshold:', reference['stem'], flush=True)
    overview.supylabel('Frequency (kHz; mel-spaced)', x=.008, fontsize=12)
    overview.supxlabel('Time within excerpt (s)', y=.012, fontsize=12)
    original.save(overview, share, '00_OVERVIEW', 250)
    summary = {}
    for model in plan['sources']:
        rows = [r for r in effects if r['model'] == model['name']]
        values = {k:sum(r[k] for r in rows) for k in ['original_foreground_pixels','common_foreground_pixels','removed_pixels','added_pixels']}
        summary[model['name']] = dict(original_threshold=model['original_threshold'], common_threshold=THRESHOLD,
            **values, fraction_of_original_foreground_removed=values['removed_pixels']/max(1,values['original_foreground_pixels']))
    base.write_json(share / 'threshold_effect.json', dict(summary=summary, per_example=effects,
        caveat='Foreground coverage changes, not accuracy measurements; these excerpts have no human masks.'))
    attribution = (Path(plan['sources'][0]['folder']) / 'share/README.md').read_text().split('## Recording attribution',1)[1]
    notes = ['# XC comparison: common threshold 0.09', '',
        'Open 00_OVERVIEW.png first. Eight individual three-column PNG/PDF comparisons also include clean inputs above predictions.',
        'Left: frozen encoder + linear. Middle: frozen encoder + transformer head. Right: fully fine-tuned encoder + linear. All use threshold 0.09.',
        'Same three models and eight excerpts as the previous grid. This does not include the newer transformer-block-only fine-tuning variant.',
        'Previous thresholds were 0.07, 0.05 and 0.09 respectively. Only the first two models therefore change; full-fine-tuning masks must be identical.',
        'Same cached continuous probabilities, source spectrograms, full-source smoothing and display. No retraining, inference or smoothing rerun; binary masks are recomputed at the shared threshold.',
        plan['postprocessing'], plan['caveat'], plan['reference'],
        'threshold_effect.json records how many foreground pixels disappeared; these counts are not accuracy measurements.', '',
        '## Recording attribution' + attribution]
    (share / 'README.md').write_text('\n'.join(notes)+'\n')
    if any(base.digest(p) != sha for p,sha in plan['protected'].items()):
        raise ValueError('source artifacts changed during rendering')
    base.write_json(share / 'provenance.json', dict(manifest_sha256=base.digest(OUT / 'manifest.json'),
        sources=plan['sources'], source_sha256=plan['protected'], common_threshold=THRESHOLD,
        no_new_inference=True, no_new_smoothing=True))
    hashes = {p.name:base.digest(p) for p in sorted(share.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'}
    (share / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
    base.write_json(OUT / 'complete.json', dict(state='complete', examples=8, files=hashes))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
