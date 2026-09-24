"""Framed NPZ inference over SSH; GPU 2 only, no benchmark storage on V100."""
import argparse
import hashlib
import io
import itertools
from pathlib import Path
import struct
import sys

import numpy as np


def receive(stream):
    header=stream.read(8)
    if not header:
        return None
    length=struct.unpack('!Q',header)[0]
    data=bytearray()
    while len(data)<length:
        chunk=stream.read(min(length-len(data),8*1024**2))
        if not chunk:
            raise EOFError('SSH inference stream ended')
        data.extend(chunk)
    with np.load(io.BytesIO(data),allow_pickle=False) as message:
        return {key:message[key] for key in message.files}


def send(stream, **arrays):
    buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays)
    data=buffer.getvalue(); stream.write(struct.pack('!Q',len(data))); stream.write(data); stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    assert hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()==args.sha256
    wire=sys.stdout.buffer
    sys.stdout=sys.stderr  # Keep library logging out of the binary protocol.
    # itertools.batched was introduced in Python 3.12; identical 3.10 fallback.
    if not hasattr(itertools,'batched'):
        def batched(iterable,n):
            iterator=iter(iterable)
            while batch:=tuple(itertools.islice(iterator,n)):
                yield batch
        itertools.batched=batched
    import torch
    from ultralytics import YOLO, __version__
    from birdsong_detect_distill.birdbox import predict_boxes
    assert __version__=='8.3.109'
    model=YOLO(args.checkpoint)
    while (request:=receive(sys.stdin.buffer)) is not None:
        with torch.inference_mode():
            boxes=predict_boxes(model,request['audio'],int(request['rate']), 'cuda:2',
                                confidence=1e-5,batch_size=8,max_det=10000)
        send(wire,boxes=boxes)


if __name__=='__main__':
    main()
