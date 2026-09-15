"""One affine projection after LayerNorm, initialized from the matched two-layer head."""
import hashlib

import torch
from torch import nn

from .pointwise_no_position import NoPositionHead

ARCHITECTURE = 'pointwise_single_linear_fused_initial_v1'


class SingleLinearHead(NoPositionHead):
    def __init__(self, dimension, hidden, height, width, patch_height, patch_width, dropout=.1, layers=0):
        if hidden != 0:
            raise ValueError('hidden=0 denotes no intermediate projection')
        super().__init__(dimension,128,height,width,patch_height,patch_width,dropout,layers)
        self.control_initial_sha256 = hashlib.sha256(b''.join(
            v.detach().cpu().numpy().tobytes() for v in self.state_dict().values())).hexdigest()
        first, second = self.project[1], self.output
        # Preserve the control's CPU RNG stream for identical minibatch ordering.
        with torch.random.fork_rng(devices=[]):
            fused = nn.Linear(dimension,patch_height*patch_width)
        with torch.no_grad():
            fused.weight.copy_(second.weight @ first.weight)
            fused.bias.copy_(second.weight @ first.bias + second.bias)
        self.norm, self.output = self.project[0], fused
        del self.project

    def forward(self, tokens, valid):
        x = self.output(self.norm(tokens)).reshape(
            len(tokens),self.height,self.width,self.patch_height,self.patch_width)
        return x.permute(0,1,3,2,4).reshape(
            len(tokens),self.height*self.patch_height,self.width*self.patch_width)


def load_heads(paths, backbone, device):
    heads = {}
    config = backbone.config
    for path in paths:
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('detector_architecture') != ARCHITECTURE or saved['head_layers'] != 0 or saved['hidden'] != 0:
            raise ValueError('not a tagged single-linear checkpoint')
        head = SingleLinearHead(config.enc_hidden_d,0,saved['height'],saved['width'],
            config.patch_height,config.patch_width,saved['dropout'],layers=0).to(device)
        head.load_state_dict(saved['head'])
        heads[path.stem] = head.eval()
    return heads
