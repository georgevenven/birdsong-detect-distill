#!/usr/bin/env python3
"""Detached, resumable axis pilot followed by xhigh and two cumulative self-reviews."""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests
from PIL import Image, ImageDraw, ImageFont

from qwen_prompt_study import OUT, PLOT, ROOT, SERVER, URL, digest, parsed_events, prepare, review_revision, score, table, verify, write

STOP=threading.Event()


def data_url(picture):
    if isinstance(picture,Image.Image):
        stream=io.BytesIO(); picture.save(stream,'PNG'); value=stream.getvalue()
    else: value=picture.read_bytes()
    return 'data:image/png;base64,'+base64.b64encode(value).decode()


def review_overlay(picture,events,numbered):
    overlay=Image.open(picture).convert('RGB'); draw=ImageDraw.Draw(overlay)
    left,top,right,bottom=PLOT; boxes=[]
    for event in events:
        x0,y0,x1,y1=event['bbox_2d']
        box=(left+x0/1000*(right-left),top+y0/1000*(bottom-top),
            left+x1/1000*(right-left),top+y1/1000*(bottom-top))
        draw.rectangle(box,outline='#ff3b30',width=4); boxes.append(box)
    if not numbered: return overlay,events
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
    occupied=[]
    for index,(x0,y0,x1,y1) in enumerate(boxes,1):
        label=str(index); bounds=draw.textbbox((0,0),label,font=font)
        width=bounds[2]-bounds[0]+10; height=bounds[3]-bounds[1]+8
        candidates=[]
        for x,y in [(x0,y0-height-3),(x0,y1+3),(x1-width,y0-height-3),
                (x1-width,y1+3),(x0,y0+3),(x1-width,y1-height-3)]:
            x=max(left,min(x,right-width)); y=max(top,min(y,bottom-height))
            candidates.append((x,y,x+width,y+height))
        def overlap(a):
            return sum(max(0,min(a[2],b[2])-max(a[0],b[0]))*
                max(0,min(a[3],b[3])-max(a[1],b[1])) for b in occupied)
        tag=min(candidates,key=overlap); occupied.append(tag)
        cx,cy=(tag[0]+tag[2])/2,(tag[1]+tag[3])/2
        anchor=min([(x0,y0),(x1,y0),(x0,y1),(x1,y1)],key=lambda p:(p[0]-cx)**2+(p[1]-cy)**2)
        draw.line((cx,cy,*anchor),fill='#ff3b30',width=2)
        draw.rectangle(tag,fill='#c21d14')
        draw.text((tag[0]+5-bounds[0],tag[1]+4-bounds[1]),label,font=font,fill='white')
    return overlay,[dict(box_id=index,**event) for index,event in enumerate(events,1)]


def request(window,condition,mode,plan,limit):
    from pathlib import Path
    previous={'self_review_1':'reasoning','self_review_2':'self_review_1'}.get(condition)
    picture=Path(window['images'][mode])
    content=[dict(type='text',text='Annotate the complete colored spectrogram. Coordinates are relative to that plot, not the surrounding canvas.'),
        dict(type='image_url',image_url={'url':data_url(picture)})]
    system=plan['system_prompt']; parent_hash=None
    if previous:
        source=OUT/'annotations'/previous/f'{window["name"]}.json'
        parent=json.loads(source.read_text()); parent_hash=digest(source)
        assert parent['manifest_sha256']==digest(OUT/'manifest.json') and parent['mode']==mode
        assert parent.get('review_amendment_sha256')==review_revision(plan,previous)
        overlay,events=review_overlay(picture,parent['raw_annotation']['events'],plan.get('numbered_review',False))
        content += [dict(type='text',text='Previous event JSON: '+json.dumps(events,separators=(',',':'))),
            dict(type='image_url',image_url={'url':data_url(overlay)})]
        system+='\n'+plan['self_review_prompt']
    thinking=not condition.startswith('direct_'); settings=plan['reasoning' if thinking else 'no_reasoning']
    offset={'self_review_1':1,'self_review_2':2}.get(condition,0)
    payload=dict(model='qwen3.8-27b-q8',messages=[dict(role='system',content=system),dict(role='user',content=content)],
        temperature=plan['temperature'],top_p=plan['top_p'],top_k=plan['top_k'],seed=window['seed']+offset,
        max_tokens=limit,reasoning_effort=settings['reasoning_effort'],reasoning_budget_tokens=settings['reasoning_budget_tokens'],
        reasoning_format='deepseek',chat_template_kwargs=dict(add_vision_id=True,enable_thinking=thinking),
        response_format=dict(type='json_schema',json_schema=dict(name='bird_vocalizations',strict=True,schema=plan['schema'])))
    return payload,parent_hash


def annotate(window,condition,mode,plan):
    saved=OUT/'annotations'/condition/f'{window["name"]}.json'
    thinking=not condition.startswith('direct_'); settings=plan['reasoning' if thinking else 'no_reasoning']
    if saved.exists():
        result=json.loads(saved.read_text())
        assert result['manifest_sha256']==digest(OUT/'manifest.json') and result['mode']==mode and result['mode_verified']
        assert result.get('review_amendment_sha256')==review_revision(plan,condition)
        _,parent_hash=request(window,condition,mode,plan,settings['max_tokens'])
        assert parent_hash==result['parent_annotation_sha256']
        return
    folder=OUT/'attempts'/condition/window['name']
    for attempt in range(3):
        if STOP.is_set(): return
        payload,parent_hash=request(window,condition,mode,plan,settings['max_tokens'] if attempt==0 else settings['retry_max_tokens'])
        stamp=time.time_ns(); started=time.time()
        audit=dict(condition=condition,mode=mode,window=window['name'],attempt=attempt,started_unix=started,
            review_amendment_sha256=review_revision(plan,condition),
            seed=payload['seed'],max_tokens=payload['max_tokens'],reasoning_effort=payload['reasoning_effort'],
            reasoning_budget_tokens=payload['reasoning_budget_tokens'],parent_annotation_sha256=parent_hash,
            request_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
        try:
            response=requests.post(URL+'/v1/chat/completions',json=payload,timeout=1200)
            audit['elapsed_seconds']=time.time()-started; audit['http_status']=response.status_code
            response.raise_for_status(); raw=response.json(); choice=raw['choices'][0]; message=choice['message']
            reasoning=message.get('reasoning_content') or message.get('reasoning') or ''; content=message.get('content') or ''
            audit.update(finish_reason=choice['finish_reason'],usage=raw.get('usage'),timings=raw.get('timings'),
                response_id=raw.get('id'),reasoning_characters=len(reasoning),content=content,
                reasoning_tokens=raw.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens'))
            verified=bool(reasoning) if thinking else not reasoning and audit['reasoning_tokens'] in (None,0)
            audit['mode_verified']=verified and '<think>' not in content and '</think>' not in content
            if not audit['mode_verified']:
                STOP.set(); raise RuntimeError('returned reasoning does not match the requested experimental mode')
            if choice['finish_reason']!='stop': raise ValueError('response ended without a complete final answer: '+choice['finish_reason'])
            result=json.loads(content); events=parsed_events(result,window)
            audit['accepted']=True; write(folder/f'{stamp}.json',audit)
            write(saved,dict(**{k:v for k,v in audit.items() if k!='content'},manifest_sha256=digest(OUT/'manifest.json'),
                raw_annotation=result,events=events))
            return
        except Exception as error:
            audit.update(accepted=False,error=str(error),elapsed_seconds=time.time()-started)
            write(folder/f'{stamp}.json',audit)
            if STOP.is_set() or attempt==2: raise
            time.sleep(2**attempt)


def stage(tasks,plan,name):
    log=OUT/'batches'/f'{name}-{time.time_ns()}.json'
    record=dict(stage=name,started_unix=time.time(),total=len(tasks),completed=0)
    write(log,record); iterator=iter(tasks)
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        running={}
        def submit():
            item=next(iterator,None)
            if item is not None:
                window,condition,mode=item
                running[pool.submit(annotate,window,condition,mode,plan)]=item
        for _ in range(plan['workers']): submit()
        try:
            while running:
                done,_=wait(running,return_when=FIRST_COMPLETED)
                for future in done:
                    window,condition,_=running.pop(future); future.result()
                    if (OUT/'annotations'/condition/f'{window["name"]}.json').exists():
                        record['completed']+=1
                        print(f'{name}: {record["completed"]}/{record["total"]} {condition} {window["name"]}',flush=True)
                    write(OUT/'status.json',dict(state='pausing' if STOP.is_set() else 'annotating',**record))
                    if not STOP.is_set(): submit()
        except BaseException:
            STOP.set(); raise
        finally:
            record['ended_unix']=time.time(); write(log,record)
    if STOP.is_set(): raise InterruptedError('paused after in-flight requests finished')


def preflight(plan,condition,mode):
    window=plan['windows'][0]; thinking=not condition.startswith('direct_')
    payload,_=request(window,condition,mode,plan,plan['reasoning' if thinking else 'no_reasoning']['max_tokens'])
    response=requests.post(URL+'/apply-template',json=payload,timeout=30); response.raise_for_status()
    prompt=response.json()['prompt']
    expected=r'<think>\s*$' if thinking else r'<think>\s*</think>\s*$'
    if not re.search(expected,prompt) or thinking and 'Reasoning effort is set to xhigh.' not in prompt:
        raise ValueError('server template does not match requested reasoning mode')
    write(OUT/'template_checks'/f'{condition}-{mode}.json',dict(thinking=thinking,mode=mode,
        suffix=prompt[-100:],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))


def run(plan):
    if (OUT/'status.json').exists() and json.loads((OUT/'status.json').read_text()).get('state')=='complete':
        print('Study already complete; nothing restarted.'); return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPUs occupied; refusing to interfere')
    try:
        if requests.get(URL+'/health',timeout=2).ok: raise ValueError('port 8080 already serves another model')
    except requests.ConnectionError: pass
    subprocess.run(['systemctl','--user','start',SERVER],check=True)
    try:
        deadline=time.monotonic()+180
        while True:
            if STOP.is_set(): raise InterruptedError('paused during server startup')
            try: ready=requests.get(URL+'/health',timeout=2).ok
            except requests.RequestException: ready=False
            if ready: break
            if time.monotonic()>deadline: raise ValueError('server startup timed out')
            time.sleep(2)
        props=requests.get(URL+'/props',timeout=10).json()
        assert props['total_slots']==16
        write(OUT/f'server_props-{time.time_ns()}.json',props)
        for mode in ['plain','axes']: preflight(plan,'direct_'+mode,mode)
        tasks=[]
        for index,window in enumerate(plan['windows']):
            modes=['plain','axes'] if index%2==0 else ['axes','plain']
            tasks.extend((window,'direct_'+mode,mode) for mode in modes)
        stage(tasks,plan,'axis_pilot')
        verify(plan)
        plain=score(plan,'direct_plain'); axes=score(plan,'direct_axes')
        decision=OUT/'axis_decision.json'
        chosen='axes' if axes['calibration']['ap']>plain['calibration']['ap'] else 'plain'
        selection=dict(mode=chosen,criterion=plan['axis_selection'],calibration_plain=plain['calibration'],
            calibration_axes=axes['calibration'],manifest_sha256=digest(OUT/'manifest.json'))
        if decision.exists(): assert json.loads(decision.read_text())==selection
        else: write(decision,selection)
        complete=['direct_plain','direct_axes']; table(complete)
        print('Axis decision (calibration only): '+json.dumps(selection),flush=True)
        for condition in plan['subsequent_conditions']:
            preflight(plan,condition,chosen)
            stage([(window,condition,chosen) for window in plan['windows']],plan,condition)
            verify(plan); score(plan,condition); complete.append(condition); table(complete)
        write(OUT/'status.json',dict(state='complete',conditions=complete,windows=250,accepted_calls=1250,selected_image_format=chosen))
        print('All five conditions complete. Original XC queue remains paused.',flush=True)
    finally:
        subprocess.run(['systemctl','--user','stop',SERVER],check=True)


if __name__=='__main__':
    os.chdir(ROOT)
    for sig in [signal.SIGINT,signal.SIGTERM]: signal.signal(sig,lambda *_:STOP.set())
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare',action='store_true'); args=parser.parse_args()
    plan=prepare()
    if args.prepare: print('Frozen 250 windows and 500 paired images.',flush=True)
    else:
        try: run(plan)
        except BaseException as error:
            write(OUT/'status.json',dict(state='paused' if isinstance(error,InterruptedError) else 'failed',error=str(error)))
            raise
