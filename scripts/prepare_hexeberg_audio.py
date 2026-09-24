"""Recover only the selected BirdSet clips; validate alignment and render native STFT."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import tarfile

import cv2
import librosa
import numpy as np
import soundfile as sf

import hexeberg_suite as suite
from birdsong_detect_distill.birdbox import spectrogram_image
from birdsong_detect_distill.benchmark_metrics import MEL_EDGES
from birdsong_detect_distill.data import FOREGROUND, load_spec_slice, read_rows


def render(encoded, row, partition, archive, member):
    name = row['recording']; root = suite.ROOT/'dataset'
    receipt = root/'provenance'/f'{name}.json'
    image = root/'images'/partition/f'{name}.png'
    labels = root/'labels'/partition/f'{name}.txt'
    source_sha = hashlib.sha256(encoded).hexdigest()
    if receipt.exists():
        saved = suite.read(receipt)
        if saved['source_sha256']!=source_sha or suite.sha(image)!=saved['image_sha256'] or suite.sha(labels)!=saved['labels_sha256']:
            raise ValueError('Cached source changed: '+name)
        return
    audio, rate = sf.read(io.BytesIO(encoded),dtype='float32',always_2d=True)
    tile = row['tile']; left, right = tile['start_timebin'],tile['end_timebin']
    if right-left!=1000:
        raise ValueError('Not a five-second teacher window')
    # Validate temporal identity against the quantized SongMAE spectrogram cache.
    mono = audio.mean(axis=1)
    if rate!=32000:
        mono = librosa.resample(mono,orig_sr=rate,target_sr=32000,res_type='soxr_hq')
    a, b = max(0,left-4),right+4
    segment = mono[a*160:min(len(mono),b*160)]
    mel = librosa.feature.melspectrogram(y=segment,sr=32000,n_fft=1024,hop_length=160,n_mels=128,fmin=20,fmax=16000,power=2.)
    observed = librosa.power_to_db(mel,ref=np.max,top_db=None)[:,left-a:right-a]
    source = row['source']
    reference = load_spec_slice(suite.REPO/'data/xcl/shards'/Path(source['shard']).name,source['start']+left,source['start']+right)
    if observed.shape!=reference.shape:
        raise ValueError('Raw audio duration differs: '+name)
    residual = observed[:,4:-4]-reference[:,4:-4]
    error = float(np.median(np.abs(residual-np.median(residual))))
    # Below power_to_db's numerical floor, correlation is undefined. Check the
    # same interior as correlation, not padding/context outside that region.
    interior_power = mel[:,left-a:right-a][:,4:-4]
    silent_match = (np.all(interior_power<=1e-10) and np.isfinite(reference).all()
        and np.ptp(reference[:,4:-4])<=1.0
        and np.max(np.abs(residual-np.median(residual)))<=1.0)
    correlation = None if silent_match else float(np.corrcoef(observed[:,4:-4].ravel(),reference[:,4:-4].ravel())[0,1])
    if not silent_match and (not np.isfinite(correlation) or correlation<.98 or error>1.0):
        raise ValueError(f'Audio/spectrogram mismatch {name}: correlation={correlation}, median error={error}')
    # Native first-channel audio, exact five-second ownership interval, one second silence.
    start, stop = round(left/200*rate),round(right/200*rate)
    clip = audio[start:stop,0]
    if round(5*rate)-len(clip)>int(np.ceil(.005*rate)):
        raise ValueError('Selected audio does not contain five complete seconds: '+name)
    actual_samples=len(clip)
    clip=np.pad(clip,(0,round(5*rate)-len(clip)))
    padded = np.pad(clip,(0,round(rate)))
    picture = spectrogram_image(padded,rate)
    for folder in [image.parent,labels.parent,receipt.parent,root/'audio']:
        folder.mkdir(parents=True,exist_ok=True)
    image_part = image.with_suffix('.pending.png')
    if not cv2.imwrite(str(image_part),picture):
        raise ValueError('Image encoding failed')
    image_part.replace(image)
    lines=[]; outside=0
    for event in row['events']:
        if event['label'] not in FOREGROUND:
            continue
        t0=max(left,event['start_timebin'])/200-left/200
        t1=min(right,event['end_timebin'])/200-left/200
        low,high=librosa.mel_to_hz(MEL_EDGES[[event['low_mel_bin'],event['high_mel_bin']]])
        low,high=max(500,float(low)),min(12000,float(high))
        if t1<=t0 or high<=low:
            outside+=1; continue
        lines.append(f'0 {(t0+t1)/12:.9f} {1-((low+high)/2-500)/11500:.9f} {(t1-t0)/6:.9f} {(high-low)/11500:.9f}')
    labels.write_text('\n'.join(lines)+ ('\n' if lines else ''))
    waveform=root/'audio'/f'{name}.wav'
    sf.write(waveform,clip,rate,subtype='FLOAT')
    suite.atomic(receipt,dict(recording=name,partition=partition,source_sha256=source_sha,archive=archive,member=member,
        native_sample_rate=rate,start_seconds=left/200,labelled_seconds=5,silence_seconds=1,
        terminal_frame_padding_samples=round(5*rate)-actual_samples,
        spectrogram_correlation=correlation,alignment_check='mel_numerical_floor' if silent_match else 'correlation',
        comparison_max_mel_power=float(interior_power.max()),
        median_centered_error_db=error,boxes=len(lines),cropped_out_boxes=outside,
        image_sha256=suite.sha(image),labels_sha256=suite.sha(labels),audio_sha256=suite.sha(waveform)))


def main():
    plan=suite.prepare(); root=suite.ROOT/'dataset'
    if (root/'summary.json').exists():
        return
    selected={}
    for key,partition in [('25000','train'),('validation','val')]:
        for row in read_rows(plan['datasets'][key]['path'],0):
            if row['recording'] in selected:
                raise ValueError('Duplicate or leaking recording')
            selected[row['recording']]=(row,partition)
    if len(selected)!=5500:
        raise ValueError('Wrong annotation budget')
    def process(item):
        name=Path(item['path']).name
        marker=suite.ROOT/'transfers'/f'{name}.complete.json'
        if marker.exists():
            return
        path=suite.ROOT/'downloads'/name
        suite.download(f'{suite.BIRDSET}/resolve/{suite.REVISION}/{item["path"]}',path,item['lfs']['oid'],item['size'])
        found=[]
        with tarfile.open(path,'r|gz') as archive:
            for member in archive:
                identifier=Path(member.name).stem
                if not member.isfile() or identifier not in selected:
                    continue
                row,partition=selected[identifier]
                render(archive.extractfile(member).read(),row,partition,name,member.name)
                found.append(identifier)
                print('Rendered',identifier,partition,flush=True)
        suite.atomic(marker,dict(sha256=item['lfs']['oid'],recordings=found))
        # Only remove this task's verified, fully processed temporary archive.
        path.unlink()
        print('Finished archive',name,'selected',len(found),flush=True)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(process,plan['archives']))
    receipts=[suite.read(root/'provenance'/f'{name}.json') for name in selected]
    summary={partition:dict(windows=sum(r['partition']==partition for r in receipts),
        boxes=sum(r['boxes'] for r in receipts if r['partition']==partition)) for partition in ['train','val']}
    (root/'dataset.yaml').write_text(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n')
    suite.atomic(root/'summary.json',dict(summary,manifest_sha256=suite.sha(suite.OUT/'manifest.json'),
        labelled_train_seconds=25000,labelled_validation_seconds=2500,receipts=receipts))


if __name__=='__main__':
    main()
