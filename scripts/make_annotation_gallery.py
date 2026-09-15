#!/usr/bin/env python3
"""Render 50 source-frozen qualitative examples using the existing 20k detector."""
import json
import hashlib
import random
from pathlib import Path

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from birdsong_detect_distill import hawaii
from birdsong_detect_distill.benchmark_data import digest, recordings, read_audio, model_audio, write_json
from birdsong_detect_distill.model import load_detector
from evaluate_2d import references, songmae_probability
from evaluate_songmae_smoothing import smooth

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
OUT = ROOT / 'results/annotation_gallery_20k_2026-09-11'
PUBLIC = ROOT / 'annotation-gallery/public'
CHECKPOINT = ROOT / 'artifacts/detector_study_2026-09-11/budget_d128_expanded20k_s0.pt'
REPORT = ROOT / 'results/qwen_teacher_powdermill/detector_study_2026-09-11/runs/budget_d128_expanded20k_s0/comparison.json'
EXTERNAL = ROOT / 'results/current_external_2026-09-09/manifest.json'
DATASETS = [('powdermill','Powdermill',RAW,False), ('wabad','WABAD',RAW,False),
    ('hawaii','Hawaii',ROOT/'data/hawaii/zenodo',False), ('xcsl','XC-AJ',RAW,True),
    ('nips4bplus','NIPS4Bplus',ROOT/'data/nips4bplus',True)]


def select():
    source = json.loads(EXTERNAL.read_text())
    selected = []
    for key,label,folder,temporal in DATASETS:
        allowed = set(source['powdermill']['evaluation'] if key=='powdermill' else source['external'][key]['recordings'])
        candidates = list(hawaii.recordings(folder) if key=='hawaii' else recordings(folder,key))
        candidates = [r for r in candidates if r['name'] in allowed and
            any(b>a and d>c for a,b,c,d in r['events'])]
        rng = random.Random(17)
        for index,record in enumerate(rng.sample(sorted(candidates,key=lambda r:r['name']),10),1):
            valid = record['events'][(record['events'][:,1]>record['events'][:,0]) & (record['events'][:,3]>record['events'][:,2])]
            event = valid[rng.randrange(len(valid))]
            selected.append(dict(id=f'{key}-{index:02d}',dataset=key,label=label,temporal_only=temporal,
                proposed_start=float(np.floor((event[0]+event[1])/10)*5),
                record={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in record.items()}))
    calibration=json.loads(REPORT.read_text())
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert saved['metrics']['train_timebins']==20000*200 and saved['seed']==0
    assert saved['hidden']==128 and saved['head_layers']==1
    protocol=json.loads(Path(calibration['protocol']).read_text()) if isinstance(calibration['protocol'],str) else calibration['protocol']
    assert digest(CHECKPOINT)==protocol['checkpoint']['checkpoint_sha256']
    return dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),
        report_sha256=digest(REPORT),external_manifest_sha256=digest(EXTERNAL),
        threshold=calibration['summary']['threshold'],backbone=saved['backbone_id'],revision=saved['backbone_revision'],
        selection='seed 17; ten distinct eligible evaluation recordings with positive references per dataset; one random human event anchors a five-second excerpt; no prediction-based selection',
        inference='complete source recording; native 5-second windows, 2.5-second stride, maximum overlap; probability Gaussian sigma=(2,3); crop only after smoothing; threshold from Powdermill',
        code_sha256={p:digest(ROOT/p) for p in ['scripts/make_annotation_gallery.py','scripts/evaluate_2d.py',
            'scripts/evaluate_songmae_smoothing.py','src/birdsong_detect_distill/model.py','src/birdsong_detect_distill/benchmark_data.py']},
        examples=selected)


def panel(path,spec,events,mask,ignored,duration,temporal,prediction):
    fig,ax=plt.subplots(figsize=(10,2.625),dpi=160)
    fig.subplots_adjust(left=.065,right=.992,bottom=.22,top=.975)
    ax.imshow(spec,origin='lower',aspect='auto',extent=(0,duration,0,128),cmap='magma',vmin=-70,vmax=0,interpolation='nearest')
    color='#ffbe66' if prediction else '#41e4fa'
    if prediction:
        if mask.any():
            ax.contour((np.arange(mask.shape[1]+2)-.5)*duration/mask.shape[1],np.arange(130)-.5,
                np.pad(mask,1),levels=[.5],colors=color,linewidths=.75)
    elif temporal:
        for start,end,_,_ in events:
            ax.axvspan(start,end,facecolor=color,alpha=.13,edgecolor='none')
            ax.vlines([start,end],0,128,colors=color,linewidth=.85)
            ax.hlines(125,start,end,colors=color,linewidth=2.5)
    else:
        for left,low,right,high in references(events):
            ax.add_patch(Rectangle((left,low),right-left,high-low,fill=False,edgecolor=color,linewidth=.85))
    for start,end in ignored:
        ax.axvspan(start,end,facecolor='#b0bac5',alpha=.3,hatch='///',edgecolor='#acbac6',linewidth=0)
    ticks=(librosa.hz_to_mel([1000,2000,4000,8000,16000])-librosa.hz_to_mel(20))*128/(librosa.hz_to_mel(16000)-librosa.hz_to_mel(20))
    ax.set_yticks(ticks,labels=['1','2','4','8','16'])
    ax.set(xlim=(0,duration),ylim=(0,128),xlabel='Time within excerpt (s)',ylabel='kHz (mel)')
    ax.tick_params(labelsize=11)
    fig.savefig(path,facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'arrays').mkdir(exist_ok=True)
    (PUBLIC/'examples').mkdir(exist_ok=True)
    manifest=OUT/'manifest.json'
    if not manifest.exists(): write_json(manifest,select())
    plan=json.loads(manifest.read_text())
    assert digest(CHECKPOINT)==plan['checkpoint_sha256'] and digest(REPORT)==plan['report_sha256']
    assert all(digest(ROOT/p)==sha for p,sha in plan['code_sha256'].items())
    device=torch.device('cuda:0')
    backbone,head,_=load_detector(CHECKPOINT,device,revision=plan['revision'])
    plt.style.use('default')
    plt.rcParams.update({'font.size':12,'axes.labelsize':11,'axes.edgecolor':'#8993a2','text.color':'#17233b','axes.labelcolor':'#40516a','xtick.color':'#40516a','ytick.color':'#40516a'})
    gallery=dict(checkpoint=CHECKPOINT.name,checkpoint_sha256=plan['checkpoint_sha256'],threshold=plan['threshold'],datasets=[])
    for key,label,_,temporal in DATASETS:
        note=('Human onset/offset labels only; frequency extent is not annotated.' if temporal else 'Human time–frequency boxes compared with the actual predicted mask contours.')
        dataset=dict(id=key,label=label,temporal_only=temporal,note=note,examples=[])
        for item in [x for x in plan['examples'] if x['dataset']==key]:
            record=item['record']; events=np.asarray(record['events'],float).reshape(-1,4)
            image_paths=[PUBLIC/'examples'/f'{item["id"]}-{kind}.png' for kind in ['truth','prediction']]
            entry_path=OUT/f'{item["id"]}.json'
            if entry_path.exists():
                entry=json.loads(entry_path.read_text())
                assert entry['manifest_sha256']==digest(manifest)
                assert all(digest(p)==entry['image_sha256'][p.name] for p in image_paths)
            else:
                wave,sr=read_audio(record); duration=len(wave)/sr
                audio,sr=model_audio(wave,sr,'songmae')
                # Preserve full-recording normalization, overlap context and smoothing.
                probabilities=smooth(songmae_probability(backbone,head,audio,device))
                fullspec=librosa.power_to_db(librosa.feature.melspectrogram(y=audio,sr=32000,n_fft=1024,
                    hop_length=160,power=2,n_mels=128,fmin=20,fmax=16000),ref=np.max,top_db=None)
                start=max(0,min(item['proposed_start'],duration-5))
                left=round(start*200); right=min(left+1000,int(np.floor(duration*200)))
                start=left/200; length=(right-left)/200
                probability=probabilities[:,left:right]; spec=fullspec[:,left:right]
                mask=probability.astype(np.float64)>=plan['threshold']
                events=events[(events[:,1]>start)&(events[:,0]<start+length)&(events[:,1]>events[:,0])&(events[:,3]>events[:,2])].copy()
                events[:,0]=np.maximum(0,events[:,0]-start); events[:,1]=np.minimum(length,events[:,1]-start)
                ignored=np.asarray([[max(0,a-start),min(length,b-start)] for a,b in record.get('ignored',[]) if b>start and a<start+length]).reshape(-1,2)
                arrays=OUT/'arrays'/f'{item["id"]}.npz'
                np.savez_compressed(arrays,spectrogram=spec,probability=probability,mask=mask,events=events,ignored=ignored)
                for path,prediction in zip(image_paths,[False,True]): panel(path,spec,events,mask,ignored,length,temporal,prediction)
                entry=dict(id=item['id'],recording=record['name'],start=start,duration=length,events=len(events),
                    ground_truth=f'/examples/{image_paths[0].name}',prediction=f'/examples/{image_paths[1].name}',
                    manifest_sha256=digest(manifest),audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                    arrays_sha256=digest(arrays),image_sha256={p.name:digest(p) for p in image_paths})
                write_json(entry_path,entry)
            dataset['examples'].append({k:entry[k] for k in ['id','recording','start','duration','events','ground_truth','prediction']})
            print(f'{item["id"]}: {entry["recording"]} {entry["start"]:.2f}s',flush=True)
        gallery['datasets'].append(dataset)
        write_json(PUBLIC/'gallery.json',gallery)
        write_json(OUT/'status.json',dict(state='rendering',completed=sum(len(d['examples']) for d in gallery['datasets']),total=50))
    assert len(gallery['datasets'])==5 and all(len(d['examples'])==10 for d in gallery['datasets'])
    write_json(OUT/'status.json',dict(state='complete',completed=50,total=50))
    print('All 50 examples rendered; no full-dataset benchmarks or training were run.',flush=True)


if __name__=='__main__': main()
