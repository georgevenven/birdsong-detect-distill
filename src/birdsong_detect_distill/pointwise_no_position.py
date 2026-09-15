"""Pointwise position ablation without modifying frozen experiment dependencies."""
import hashlib

import torch

from .model import DenseHead

ARCHITECTURE = 'pointwise_no_position_v1'


class NoPositionHead(DenseHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.block is not None or self.extra_blocks:
            raise ValueError('position ablation requires zero attention layers')
        # Initialize identically to controls, then remove only the two positional parameters.
        self.control_initial_sha256 = hashlib.sha256(b''.join(
            value.detach().cpu().numpy().tobytes() for value in self.state_dict().values())).hexdigest()
        del self.frequency, self.time

    def forward(self, tokens, valid):
        x = self.output(self.project(tokens)).reshape(
            len(tokens), self.height, self.width, self.patch_height, self.patch_width)
        return x.permute(0, 1, 3, 2, 4).reshape(
            len(tokens), self.height*self.patch_height, self.width*self.patch_width)


def load_heads(paths, backbone, device):
    heads = {}
    config = backbone.config
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('detector_architecture') != ARCHITECTURE or saved['head_layers'] != 0:
            raise ValueError('not a tagged no-position pointwise checkpoint')
        head = NoPositionHead(config.enc_hidden_d, saved['hidden'], saved['height'], saved['width'],
            config.patch_height, config.patch_width, saved['dropout'], layers=0).to(device)
        head.load_state_dict(saved['head'])
        heads[path.stem] = head.eval()
    return heads
