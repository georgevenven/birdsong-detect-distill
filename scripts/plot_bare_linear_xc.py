#!/usr/bin/env python3
"""Eight unseen XC examples: native spectrograms and the actual post-processed masks."""
import csv
import hashlib
import json
import random
from pathlib import Path

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from birdsong_detect_distill.benchmark_data import digest, write_json
from birdsong_detect_distill.data import load_spec_slice
from birdsong_detect_distill.model import load_backbone
from birdsong_detect_distill.pointwise_bare_linear import ARCHITECTURE, load_heads
from evaluate_songmae_smoothing import smooth
from prepare_detector_study import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/bare_linear_xc_heldout_2026-09-14'
STUDY = ROOT / 'results/qwen_teacher_powdermill/pointwise_bare_linear_2000train_800val_2026-09-14'
XC = Path('/home/george-vengrovski/Documents/SongMAE/data/XCL_val')
QUEUE = ROOT / 'data/annotations/xcl/powdermill_protocol_50000s_2026-09-14'


def prepare():
    path = OUT / 'manifest.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if any(digest(p)!=sha for p,sha in plan['protected'].items()):
            raise ValueError('a frozen figure dependency changed')
        return plan
    study = json.loads((STUDY / 'manifest.json').read_text())
    verify(study)
    report_path = STUDY / 'runs/bare_linear_d0_new2k_s0/comparison.json'
    report = json.loads(report_path.read_text())
    saved = report['protocol']['checkpoint']
    if saved['seed'] != 0 or saved['detector_architecture'] != ARCHITECTURE or digest(saved['checkpoint']) != saved['checkpoint_sha256']:
        raise ValueError('unexpected or changed checkpoint')
    with (XC / 'shards/index.tsv').open() as stream:
        rows = list(csv.DictReader(stream,delimiter='\t'))
    train_index = ROOT / 'data/xcl/shards/index.tsv'
    with train_index.open() as stream:
        all_xcl_train = {r['name'] for r in csv.DictReader(stream,delimiter='\t')}
    if {r['name'] for r in rows} & all_xcl_train:
        raise ValueError('XCL train/validation source IDs overlap')
    queue = json.loads((QUEUE / 'manifest.json').read_text())
    exclusion = json.loads((QUEUE / 'selection.json').read_text())
    forbidden = (set(saved['training_recording_ids']) | set(saved['validation_recording_ids']) |
        {w['tile'][0] for w in queue['windows']} | set(exclusion['exclusions']['xcaj']['xc_ids']))
    metadata_path = ROOT / 'data/annotations/xcl/XCL_train_annotations.json'
    metadata = {Path(r['recording']['filename']).stem:r['recording'] for r in
        json.loads(metadata_path.read_text(),parse_constant=lambda _:None)['recordings']}
    pool = [r for r in rows if 2001 <= int(r['end'])-int(r['start']) <= 24001 and r['name'] not in forbidden]
    rng = random.Random(17)
    rng.shuffle(pool)
    examples, species, cells = [], set(), set()
    for row in pool:
        meta = metadata[row['name']]
        lat, lon = meta.get('lat'), meta.get('long')
        cell = None if lat is None or lon is None else (int(lat//10),int(lon//10))
        if meta['ebird_code'] in species or (cell is not None and cell in cells):
            continue
        start, end = int(row['start']), int(row['end'])
        offset = rng.randrange((end-start-1)//1000)*1000
        examples.append(dict(id=f'{len(examples)+1:02d}',recording=row['name'],shard=row['shard'],
            source_start=start,source_end=end,excerpt_frame=offset,metadata=meta))
        species.add(meta['ebird_code']); cells.add(cell)
        if len(examples)==8:
            break
    if len(examples)!=8:
        raise ValueError('not enough diverse held-out examples')
    protected = {str(p):digest(p) for p in [XC / 'shards/index.tsv',XC / 'audio_params.json',train_index,
        metadata_path,QUEUE / 'manifest.json',QUEUE / 'selection.json',STUDY / 'manifest.json',report_path,
        Path(saved['checkpoint']),Path(__file__),ROOT / 'src/birdsong_detect_distill/data.py',
        ROOT / 'src/birdsong_detect_distill/pointwise_bare_linear.py',ROOT / 'scripts/evaluate_songmae_smoothing.py']}
    protected.update(study['code_sha256'])
    for e in examples:
        p = XC / 'shards' / Path(e['shard']).with_suffix('.txt')
        protected[str(p)] = digest(p)
    plan = dict(checkpoint=saved['checkpoint'],backbone=saved['backbone_id'],revision=saved['backbone_revision'],
        seed=0,threshold=report['summary']['threshold'],training_seconds=2000,
        checkpoint_choice='fixed seed 0, not selected by Powdermill score or example appearance',
        selection='seed-17 shuffled XCL validation recordings lasting 10–120 s; distinct focal species/geographic cells; random aligned 5 s excerpt before inference',
        exclusions='disjoint from entire XCL training index, detector train/validation IDs, 50k teacher queue and XC-AJ; source-ID audit, not cross-ID acoustic deduplication',
        reference='No human time-frequency ground truth; top panels are unannotated input spectrograms',
        inference='dequantized native 128-mel/5-ms shards; backbone training mean/std, not validation mean/std; full source, 5 s windows, 2.5 s stride, maximum overlap, batch 4, float16 autocast',
        postprocessing='sigmoid -> full-source probability Gaussian sigma (2 mel,3 frames), reflect/truncate 4 -> fixed Powdermill-calibrated threshold; crop afterward; no morphology',
        display='magma, -70 to 0 dB relative to full-source maximum; display shift does not alter model inputs',
        examples=examples,protected=protected)
    write_json(path,plan)
    return plan


@torch.inference_mode()
def predict(raw, backbone, head, device):
    spec = (raw-backbone.config.audio_mean)/backbone.config.audio_std
    width, hop = 1000, 500
    starts = list(range(0,max(1,spec.shape[1]-width+hop),hop))
    if starts[-1]+width < spec.shape[1]:
        starts.append(starts[-1]+hop)
    probability = np.zeros_like(raw,dtype=np.float32)
    for batch_start in range(0,len(starts),4):
        batch = starts[batch_start:batch_start+4]
        lengths = [min(width,spec.shape[1]-start) for start in batch]
        values = np.stack([np.pad(spec[:,start:start+size],((0,0),(0,width-size))) for start,size in zip(batch,lengths)])
        valid = torch.tensor(lengths,device=device)
        with torch.autocast('cuda',dtype=torch.float16):
            tokens = backbone(input_values=torch.from_numpy(values)[:,None].to(device),valid_timebins=valid).last_hidden_state
            predictions = head(tokens,valid).sigmoid().float().cpu().numpy()
        for start,size,prediction in zip(batch,lengths,predictions):
            probability[:,start:start+size] = np.maximum(probability[:,start:start+size],prediction[:,:size])
    if not np.isfinite(probability).all():
        raise ValueError('nonfinite predictions')
    return probability


def panels(axes, example, raw, mask, overview=False):
    color = '#00D5FF'
    for ax in axes:
        ax.imshow(raw,origin='lower',aspect='auto',extent=(0,5,0,128),cmap='magma',vmin=-70,vmax=0,
            interpolation='nearest',rasterized=True)
        ax.set(xlim=(0,5),ylim=(0,128))
    start = example['excerpt_frame']/200
    top = f"{example['recording']} · {start:g}–{start+5:g} s\n" if overview else ''
    axes[0].set_title(top+'(a) Clean spectrogram',pad=6)
    axes[1].set_title('(b) Linear detector, post-processed',pad=6)
    if mask.any():
        axes[1].contour((np.arange(1002)-.5)/200,np.arange(130)-.5,np.pad(mask,1),
            levels=[.5],colors=color,linewidths=.65)
    ticks = (librosa.hz_to_mel([1000,2000,4000,8000,16000])-librosa.hz_to_mel(20))*128/(
        librosa.hz_to_mel(16000)-librosa.hz_to_mel(20))
    for ax in axes:
        ax.set_yticks(ticks,labels=['1','2','4','8','16'])
        ax.set_xticks(range(6))
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel('Time within excerpt (s)')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan = prepare()
    if (OUT / 'complete.json').exists():
        print('Already rendered:',OUT); return
    if torch.cuda.mem_get_info(0)[0] < 20*1024**3:
        raise ValueError('GPU occupied; preserve other work')
    checkpoint, device = Path(plan['checkpoint']),torch.device('cuda:0')
    backbone = load_backbone(plan['backbone'],device,revision=plan['revision'])
    head = load_heads([checkpoint],backbone,device)[checkpoint.stem]
    share, cache = OUT / 'share',OUT / 'arrays'
    share.mkdir(exist_ok=True); cache.mkdir(exist_ok=True)
    plt.style.use('default')
    plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42})
    overview, overview_axes = plt.subplots(4,4,figsize=(14,9.5))
    overview.subplots_adjust(left=.045,right=.99,bottom=.055,top=.91,hspace=.85,wspace=.28)
    overview.suptitle(f"Held-out XC · SongMAE-Large + linear head · 2,000 s training\nProbability smoothing + Powdermill threshold {plan['threshold']:.2f}; cyan = predicted foreground",fontsize=15)
    entries = []
    for i,e in enumerate(plan['examples']):
        raw = load_spec_slice(XC / 'shards' / e['shard'],e['source_start'],e['source_end'])
        if raw.shape[0]!=128 or not np.isfinite(raw).all():
            raise ValueError('invalid held-out spectrogram')
        probability = predict(raw,backbone,head,device)
        smoothed = smooth(probability)
        fullmask = smoothed.astype(np.float64)>=plan['threshold']
        left, right = e['excerpt_frame'],e['excerpt_frame']+1000
        display, mask = (raw-float(raw.max()))[:,left:right],fullmask[:,left:right]
        name = f"{e['id']}_{e['recording']}_{left//200}-{right//200}s"
        np.savez_compressed(cache / f'{name}.npz',spectrogram_db=display,raw_probability=probability[:,left:right],
            smoothed_probability=smoothed[:,left:right],mask=mask)
        fig,axes = plt.subplots(2,1,figsize=(3.5,3.5),sharex=True,sharey=True)
        fig.subplots_adjust(left=.17,right=.98,bottom=.14,top=.83,hspace=.55)
        fig.suptitle(f"{e['recording']} · {left/200:g}–{right/200:g} s",fontsize=11,y=.98)
        panels(axes,e,display,mask)
        fig.text(.025,.49,'Frequency (kHz; mel-spaced)',rotation=90,va='center',fontsize=9)
        for ext in ['png','pdf']:
            fig.savefig(share / f'{name}.{ext}',dpi=600,facecolor='white')
        plt.close(fig)
        row,col = (i//4)*2,i%4
        panels([overview_axes[row,col],overview_axes[row+1,col]],e,display,mask,True)
        entries.append(dict(**e,stem=name,full_source_spectrogram_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            full_probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
            foreground_fraction=float(mask.mean()),arrays_sha256=digest(cache / f'{name}.npz')))
        print(name,'foreground fraction',round(float(mask.mean()),4),flush=True)
    for ext in ['png','pdf']:
        overview.savefig(share / f'00_OVERVIEW.{ext}',dpi=250,facecolor='white')
    plt.close(overview)
    write_json(OUT / 'examples.json',dict(manifest_sha256=digest(OUT / 'manifest.json'),examples=entries))
    notes = ['# Held-out XC: bare-linear SongMAE detector','',
        'Open 00_OVERVIEW.png first. Each individual PNG/PDF is square, 3.5 inches, with 600-dpi PNG output.',
        'Top: clean spectrogram, NOT human ground truth. Bottom: cyan contours of the actual binary prediction.',
        f"Frozen SongMAE-Large + Linear(768,32), 24,608 head parameters; seed 0, 2,000 s training. Threshold {plan['threshold']:.2f} selected on Powdermill Recordings 2–4, never on these XC examples.",
        'Post-processing: sigmoid probabilities, Gaussian smoothing sigma=2 mel bins and 3 frames (15 ms), then one threshold. No hysteresis, closing, component removal, hole filling or cosmetic contour smoothing.',
        'Full-source 5-second inference windows overlap by 2.5 seconds and combine by maximum. Smooth before cropping the displayed five-second excerpt. Native dequantized XC spectrograms use the backbone training normalization.',
        plan['selection']+'.',plan['exclusions']+'.',
        'These are qualitative examples, not a localization benchmark. Focal species metadata does not verify the species inside an excerpt.','',
        '## Recording attribution','']
    for e in entries:
        m=e['metadata'];notes.append(f"- {e['stem']}: {m['recordist']}; https://xeno-canto.org/{e['recording'][2:]}; license https:{m['license']}; focal species code {m['ebird_code']}; coordinates {m.get('lat')}, {m.get('long')}.")
    (share / 'README.md').write_text('\n'.join(notes)+'\n')
    write_json(share / 'provenance.json',dict(checkpoint=checkpoint.name,checkpoint_sha256=digest(checkpoint),
        threshold=plan['threshold'],selection=plan['selection'],examples=entries))
    hashes = {p.name:digest(p) for p in sorted(share.iterdir()) if p.is_file() and p.name!='SHA256SUMS'}
    (share / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
    write_json(OUT / 'complete.json',dict(state='complete',examples=8,files=hashes))


if __name__=='__main__':
    main()
