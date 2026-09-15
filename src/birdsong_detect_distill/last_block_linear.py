"""Bare linear detector with an explicitly restored final SongMAE encoder block."""
import torch

from .pointwise_bare_linear import BareLinearHead

ARCHITECTURE = 'songmae_last_encoder_block_bare_linear_v1'
BLOCK = 'songmae.encoder.layers.11'


def last_block(backbone):
    if len(backbone.songmae.encoder.layers) != 12 or backbone.config.enc_hidden_d != 768:
        raise ValueError('this experiment requires the 12-block SongMAE-Large encoder')
    return backbone.get_submodule(BLOCK)


def load_heads(paths, backbone, device):
    if len(paths) != 1:
        raise ValueError('different fine-tuned checkpoints require separate backbone instances')
    path = paths[0]
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('detector_architecture') != ARCHITECTURE or saved.get('unfrozen_module') != BLOCK
            or saved['backbone_revision'] != backbone.config._commit_hash):
        raise ValueError('incompatible final-block checkpoint/backbone')
    last_block(backbone).load_state_dict(saved['last_block'], strict=True)
    backbone.requires_grad_(False).eval()
    config = backbone.config
    head = BareLinearHead(config.enc_hidden_d, 0, saved['height'], saved['width'],
        config.patch_height, config.patch_width, saved['dropout'], layers=0).to(device)
    head.load_state_dict(saved['head'], strict=True)
    return {path.stem: head.eval()}
