"""A single affine projection of raw SongMAE output tokens."""
import hashlib

import torch

from .pointwise_single_linear import SingleLinearHead

ARCHITECTURE = 'pointwise_bare_linear_v1'


class BareLinearHead(SingleLinearHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_initial_sha256 = hashlib.sha256(b''.join(
            v.detach().cpu().numpy().tobytes() for v in self.state_dict().values())).hexdigest()
        del self.norm

    def forward(self, tokens, valid):
        x = self.output(tokens).reshape(
            len(tokens),self.height,self.width,self.patch_height,self.patch_width)
        return x.permute(0,1,3,2,4).reshape(
            len(tokens),self.height*self.patch_height,self.width*self.patch_width)


def load_heads(paths, backbone, device):
    heads = {}
    config = backbone.config
    for path in paths:
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('detector_architecture') != ARCHITECTURE or saved.get('layer_norm') is not False:
            raise ValueError('not a tagged bare-linear checkpoint')
        head = BareLinearHead(config.enc_hidden_d,saved['hidden'],saved['height'],saved['width'],
            config.patch_height,config.patch_width,saved['dropout'],saved['head_layers']).to(device)
        head.load_state_dict(saved['head'])
        heads[path.stem] = head.eval()
    return heads
