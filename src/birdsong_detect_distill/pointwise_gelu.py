"""Isolated GELU head; existing checkpoint loaders and annotation code stay unchanged."""
import torch
from torch import nn

from .model import DenseHead

ARCHITECTURE = 'pointwise_gelu_after_position_v1'


class GeluPointwiseHead(DenseHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.block is not None or self.extra_blocks:
            raise ValueError('GELU ablation requires zero attention layers')
        # No new weights or RNG consumption. Distinct keys prevent silent loading as a linear head.
        self.output = nn.Sequential(nn.GELU(), self.output)


def load_heads(paths, backbone, device):
    heads = {}
    config = backbone.config
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('detector_architecture') != ARCHITECTURE or saved['head_layers'] != 0:
            raise ValueError('not a tagged GELU pointwise checkpoint')
        head = GeluPointwiseHead(config.enc_hidden_d, saved['hidden'], saved['height'], saved['width'],
            config.patch_height, config.patch_width, saved['dropout'], layers=0).to(device)
        head.load_state_dict(saved['head'])
        heads[path.stem] = head.eval()
    return heads
