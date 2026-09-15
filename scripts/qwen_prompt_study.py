"""Frozen prompts, paired coordinate images, and scoring for the 250-window study."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from birdsong_detect_distill.benchmark_data import digest, recordings, write_json
from birdsong_detect_distill.benchmark_metrics import THRESHOLDS, area, summarize
from birdsong_detect_distill.qwen import context, image, map_final, read_tiles, split_tiles
from evaluate_stage_pipeline import teacher_probability
from evaluate_songmae_smoothing import truth_and_coverage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/qwen_teacher_powdermill/prompt_250_2026-09-11'
SPEC = ROOT/'data/powdermill/qwen'
RAW = Path('/home/george-vengrovski/Documents/SongMAE/data/birdcode/raw')
SERVER = 'birdsong-qwen-prompt-server-20260911.service'
URL = 'http://127.0.0.1:8080'
PLOT = (128,56,2176,568)
CANVAS = (2240,704)
SYSTEM = '''You are an expert bioacoustic spectrogram annotator. Detect and tightly localize bird vocalizations using only the visible spectrogram; no audio is available.
Use one label, bird_vocalization, for all bird sounds: individual songs and calls, simultaneous birds, and inseparable choruses. Do not assign species, individual identities, noise classes, or a separate uncertain class.
Include faint or partly obscured sounds when their visible structure supports a bird origin. For ambiguous candidates, use your best judgment: include a region when there is credible visual evidence of a bird vocalization, and exclude regions better explained by insects, other non-avian sounds, or environmental or human-made noise. If an included candidate is ambiguous, reflect that uncertainty in its confidence. Faintness alone is not a reason to omit a call.
Draw a tight box around each temporally continuous vocal element or tightly connected phrase, including its visible harmonics. Separate repetitions across genuine silence. Give separable simultaneous vocalizations separate boxes. For a dense chorus that cannot be separated reliably, use a box enclosing the inseparable vocal activity and split it at genuine silence. Label these chorus boxes bird_vocalization too. Do not duplicate the same activity as both individual boxes and a chorus box.
Coordinates refer ONLY to the colored spectrogram rectangle, excluding its surrounding margin, tick marks, and text. The plot's top-left corner is (x=0,y=0) and its bottom-right corner is (x=1000,y=1000). Time increases left to right; frequency decreases top to bottom. Return integer bbox_2d coordinates in the order [x_min,y_min,x_max,y_max], with 0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000. Coordinate ticks, if present, use this same plot-relative scale; they are not seconds or hertz.
Confidence is your estimated probability that the boxed region contains a bird vocalization rather than non-bird sound. It is not a measure of loudness or boundary precision. Return only the required JSON: an events list containing label, bbox_2d, and confidence for every detection. Return an empty events list when no bird vocalization is supported. Do not include summaries or additional fields.'''
SELF_REVIEW = '''Reinspect the entire clean spectrogram independently before checking the previous boxes. The second image overlays those boxes in red, and their exact coordinates are supplied as JSON. Keep supported detections; add missed vocalizations, remove unsupported detections or duplicates, and correct boundaries and confidence where the evidence warrants it. A change is not required: retain the previous result if it is already correct. Return the complete updated event list, not a list of edits. Use the same bird_vocalization label and plot-relative coordinate convention.'''
SCHEMA = dict(type='object',properties={'events':dict(type='array',maxItems=64,items=dict(type='object',
    properties={'label':{'type':'string','enum':['bird_vocalization']},
        'bbox_2d':dict(type='array',items=dict(type='integer',minimum=0,maximum=1000),minItems=4,maxItems=4),
        'confidence':dict(type='number',minimum=0,maximum=1)},
    required=['label','bbox_2d','confidence'],additionalProperties=False))},required=['events'],additionalProperties=False)


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def verify(plan):
    expected=dict(plan['protected'])
    amendment=OUT/'review_amendment.json'
    if amendment.exists():
        change=json.loads(amendment.read_text())
        assert change['base_manifest_sha256']==digest(OUT/'manifest.json')
        assert plan.get('review_amendment_sha256',digest(amendment))==digest(amendment)
        for path,revision in change['code_replacements'].items():
            assert expected[path]==revision['before']==digest(revision['snapshot'])
            expected[path]=revision['after']
        expected.update(change['additional_inputs'])
    for path,sha in expected.items():
        if digest(path)!=sha: raise ValueError(f'frozen input changed: {path}')


def review_revision(plan,condition):
    return plan.get('review_amendment_sha256') if condition.startswith('self_review_') else None


def decorate(picture,axes):
    canvas=Image.new('RGB',CANVAS,'white'); canvas.paste(picture,PLOT[:2])
    draw=ImageDraw.Draw(canvas)
    left,top,right,bottom=PLOT
    draw.rectangle((left-1,top-1,right,bottom),outline='#202c3b',width=1)
    if axes:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
        for value in range(0,1001,100):
            x=left+value/1000*(right-left); y=top+value/1000*(bottom-top)
            draw.line((x,bottom,x,bottom+10),fill='#202c3b',width=2)
            draw.text((x,bottom+15),str(value),font=font,fill='#202c3b',anchor='mt')
            draw.line((left-10,y,left-1,y),fill='#202c3b',width=2)
            draw.text((left-17,y),str(value),font=font,fill='#202c3b',anchor='rm')
        draw.text(((left+right)/2,bottom+65),'x (0–1000; left → right)',font=font,fill='#202c3b',anchor='mt')
        draw.text((left,18),'y: 0 at top → 1000 at bottom',font=font,fill='#202c3b')
    assert np.array_equal(np.asarray(canvas)[top:bottom,left:right],np.asarray(picture))
    return canvas


def prepare():
    path=OUT/'manifest.json'
    if path.exists():
        plan=json.loads(path.read_text()); verify(plan)
        amendment=OUT/'review_amendment.json'
        if amendment.exists():
            change=json.loads(amendment.read_text())
            plan.update(self_review_prompt=change['self_review_prompt'],numbered_review=True,
                review_amendment_sha256=digest(amendment))
        return plan
    source=SPEC/'recordings.jsonl'
    tiles=split_tiles(read_tiles(source),1000)
    pool=defaultdict(list)
    for tile in tiles:
        if tile[-1]-tile[-2]==1000: pool[tile[0]].append(tile)
    rng=random.Random(17); selected=[]
    for partition,total in [('calibration',100),('evaluation',150)]:
        names=sorted(n for n in pool if n.startswith('Recording_1_')==(partition=='evaluation'))
        rng.shuffle(names)
        for i,name in enumerate(names):
            count=total//len(names)+(i<total%len(names))
            selected.extend((partition,tile) for tile in rng.sample(pool[name],count))
    rng.shuffle(selected)
    protected={str(source):digest(source),str(SPEC/'audio_params.json'):digest(SPEC/'audio_params.json')}
    windows=[]
    for index,(partition,tile) in enumerate(selected):
        name=f'{tile[0]}_{tile[-2]}_{tile[-1]}'
        spec,_,_=context(SPEC,tile,tile[-2],tile[-1]); clean=image(spec)
        pictures={}
        for mode in ['plain','axes']:
            picture=OUT/'images'/mode/f'{name}.png'; picture.parent.mkdir(parents=True,exist_ok=True)
            if picture.exists(): raise ValueError('partial image preparation without manifest; inspect before replacing')
            decorate(clean,mode=='axes').save(picture)
            pictures[mode]=str(picture); protected[str(picture)]=digest(picture)
        windows.append(dict(name=name,tile=list(tile),partition=partition,seed=17+index*31,images=pictures))
    for item in ['scripts/qwen_prompt_study.py','scripts/run_qwen_prompt_study.py','src/birdsong_detect_distill/qwen.py',
            'src/birdsong_detect_distill/benchmark_metrics.py','src/birdsong_detect_distill/benchmark_data.py',
            'scripts/evaluate_stage_pipeline.py','scripts/evaluate_songmae_smoothing.py']:
        protected[str(ROOT/item)]=digest(ROOT/item)
    protected[str(RAW/'powdermill/annotation_Files.zip')]=digest(RAW/'powdermill/annotation_Files.zip')
    plan=dict(windows=windows,protected=protected,system_prompt=SYSTEM,self_review_prompt=SELF_REVIEW,schema=SCHEMA,
        canvas=list(CANVAS),plot_rectangle_pixels=list(PLOT),label='bird_vocalization',
        sampling='seed 17; equal-as-possible window counts per five-minute segment; all 77 segments represented; 150 Recording_1, 100 Recordings_2–4; no selection using references or predictions',
        initial_conditions=['direct_plain','direct_axes'],subsequent_conditions=['reasoning','self_review_1','self_review_2'],
        axis_selection='higher calibration segment-macro pixel AP; exact ties use plain; Recording_1 is not consulted',
        workers=16,context_per_slot=16384,temperature=.5,top_p=.95,top_k=20,
        no_reasoning=dict(reasoning_effort='none',reasoning_budget_tokens=0,max_tokens=4096,retry_max_tokens=6144),
        reasoning=dict(reasoning_effort='xhigh',reasoning_budget_tokens=4096,max_tokens=8192,retry_max_tokens=10240),
        threshold_grid=THRESHOLDS.tolist(),allowed_operating_thresholds='0.01–1.00; excludes zero because zero scores outside boxes are not detections',
        metrics='segment-macro continuous pixel AP and 2D IoU; pooled precision/recall; independently calibrated on the same 100 windows; no smoothing',
        caveat='Exploratory development subset, one seed per window. Axis selection uses calibration AP; not a held-out test or downstream student ablation. Confidence wording and label taxonomy differ from historical experiments.')
    write(path,plan); return plan


def parsed_events(result,window):
    if not isinstance(result,dict) or set(result)!={'events'} or not isinstance(result['events'],list) or len(result['events'])>64:
        raise ValueError('invalid event-list schema')
    for event in result['events']:
        if not isinstance(event,dict) or set(event)!={'label','bbox_2d','confidence'}: raise ValueError('unexpected event fields')
        box=event['bbox_2d']; c=event['confidence']
        if (event['label']!='bird_vocalization' or not isinstance(box,list) or len(box)!=4
                or any(type(v)!=int or not 0<=v<=1000 for v in box) or box[0]>=box[2] or box[1]>=box[3]
                or type(c) not in (int,float) or not np.isfinite(c) or not 0<=c<=1):
            raise ValueError('invalid label, non-positive box, or confidence')
    left,right=window['tile'][-2:]
    mapped=map_final(result['events'],left,right,left,right,128,5)
    # Compatibility only: all raw outputs remain the single bird_vocalization class.
    return [{**event,'label':'target_vocalization'} for event in mapped]


def score(plan,condition):
    path=OUT/'scores'/f'{condition}.json'
    if path.exists():
        result=json.loads(path.read_text())
        assert result['manifest_sha256']==digest(OUT/'manifest.json')
        assert result.get('review_amendment_sha256')==review_revision(plan,condition)
        for file,sha in result['annotation_sha256'].items(): assert digest(file)==sha
        return result
    inventory={r['name']:r for r in recordings(RAW,'powdermill')}
    grouped=defaultdict(list)
    for window in plan['windows']: grouped[window['tile'][0]].append(window)
    parts={'calibration':[],'evaluation':[]}; hashes={}
    for name,windows in sorted(grouped.items()):
        width=windows[0]['tile'][3]-windows[0]['tile'][2]
        truth,kept=truth_and_coverage(inventory[name],width,[[w['tile'][-2]/200,w['tile'][-1]/200] for w in windows])
        assert kept.sum()==len(windows)*1000
        rows=[]
        for window in windows:
            file=OUT/'annotations'/condition/f'{window["name"]}.json'
            annotation=json.loads(file.read_text()); hashes[str(file)]=digest(file)
            assert annotation['manifest_sha256']==digest(OUT/'manifest.json')
            assert annotation.get('review_amendment_sha256')==review_revision(plan,condition)
            rows.append({'variants':{'selected':annotation}})
        probability=teacher_probability(rows,'selected',width)
        parts[windows[0]['partition']].append(dict(name=name,seconds=len(windows)*5,
            area={'full':area(probability[:,kept],truth[:,kept])}))
    mean_curve=np.mean([row['area']['full']['iou_curve'] for row in parts['calibration']],axis=0)
    threshold=1+int(np.argmax(mean_curve[1:]))
    attempts=[json.loads(p.read_text()) for p in (OUT/'attempts'/condition).glob('*/*.json')]
    completed=[a for a in attempts if a.get('accepted')]
    timing=dict(attempts=len(attempts),accepted=len(completed),
        output_tokens=sum((a.get('usage') or {}).get('completion_tokens',0) for a in attempts),
        mean_request_seconds=float(np.mean([a['elapsed_seconds'] for a in completed])),
        mean_output_tokens=float(np.mean([(a.get('usage') or {}).get('completion_tokens',0) for a in completed])),
        generated_reasoning_characters=sum(a.get('reasoning_characters',0) for a in attempts),
        note='Axis conditions are interleaved in one batch; use batches/ for wall time, not summed request latencies.')
    result=dict(condition=condition,manifest_sha256=digest(OUT/'manifest.json'),annotation_sha256=hashes,timing=timing,
        review_amendment_sha256=review_revision(plan,condition),
        threshold_index=threshold,calibration=summarize(parts['calibration'],{'full':threshold})['full'],
        evaluation=summarize(parts['evaluation'],{'full':threshold})['full'],
        all_boxes_evaluation=summarize(parts['evaluation'],{'full':1})['full'],per_recording=parts)
    write(path,result); return result


def table(conditions):
    rows=['# Simplified Qwen prompt: matched 250-window study','',
        '| Configuration | Pixel AP | 2D IoU | Pixel precision | Pixel recall | Threshold |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in conditions:
        result=json.loads((OUT/'scores'/f'{name}.json').read_text())['evaluation']
        rows.append(f'| {name} | {result["ap"]:.4f} | {result["iou"]:.4f} | {result["pooled_precision"]:.4f} | {result["pooled_recall"]:.4f} | {result["threshold"]:.2f} |')
    rows+=['','Report: 150 windows (750 s) from Recording_1. Calibration: 100 windows (500 s) from Recordings_2–4.',
        'All 77 five-minute segments represented. Same intervals for every condition; no frequency/time smoothing.',
        'Axis selection uses calibration AP only. Reasoning and self-reviews use the selected image format.',
        'These are exploratory teacher results, not student results or full-Powdermill table replacements.']
    (OUT/'comparison.md').write_text('\n'.join(rows)+'\n')
