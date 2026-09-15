#!/usr/bin/env python3
"""Re-space the overview from verified cached predictions; no inference changes."""
import json

from plot_bare_linear_xc import OUT, digest, np, panels, plt, prepare, write_json


def main():
    plan = prepare()
    share = OUT / 'share'
    complete = json.loads((OUT / 'complete.json').read_text())
    if any(digest(share / name) != sha for name, sha in complete['files'].items()):
        raise ValueError('rendered outputs changed')
    entries = json.loads((OUT / 'examples.json').read_text())['examples']
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(4, 4, figsize=(14, 9.5))
    fig.subplots_adjust(left=.045, right=.99, bottom=.055, top=.87, hspace=.85, wspace=.28)
    fig.suptitle('Held-out XC · SongMAE-Large + linear head · 2,000 s training\n'
        f"Probability smoothing + Powdermill threshold {plan['threshold']:.2f}; cyan = predicted foreground",
        fontsize=14, y=.99)
    for i, entry in enumerate(entries):
        path = OUT / 'arrays' / f"{entry['stem']}.npz"
        if digest(path) != entry['arrays_sha256']:
            raise ValueError('cached predictions changed')
        with np.load(path) as cached:
            row, col = (i // 4) * 2, i % 4
            panels([axes[row, col], axes[row + 1, col]], entry,
                cached['spectrogram_db'], cached['mask'], True)
    for ext in ['png', 'pdf']:
        fig.savefig(share / f'00_OVERVIEW.{ext}', dpi=250, facecolor='white')
    plt.close(fig)
    complete['files'] = {name: digest(share / name) for name in complete['files']}
    complete['overview_layout'] = 'Extra header spacing; unchanged cached masks and individual figures'
    write_json(OUT / 'complete.json', complete)
    (share / 'SHA256SUMS').write_text(''.join(
        f'{sha}  {name}\n' for name, sha in complete['files'].items()))


if __name__ == '__main__':
    main()
