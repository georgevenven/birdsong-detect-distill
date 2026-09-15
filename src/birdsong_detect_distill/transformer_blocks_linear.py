"""Train/restore the twelve transformer blocks, leaving the SongMAE front end fixed."""
import torch

from .pointwise_bare_linear import BareLinearHead

ARCHITECTURE = 'songmae_transformer_blocks_bare_linear_v1'
PREFIX = 'songmae.encoder.'


def encoder_key(name):
    return name.startswith(PREFIX)


def encoder_state(backbone):
    return {k:v for k,v in backbone.state_dict().items() if encoder_key(k)}


def restore_encoder(backbone, state):
    if set(state) != set(encoder_state(backbone)):
        raise ValueError('checkpoint must contain all transformer-block weights and no front-end weights')
    backbone.songmae.encoder.load_state_dict({k[len(PREFIX):]:v for k,v in state.items()}, strict=True)


def load_heads(paths, backbone, device):
    if len(paths) != 1:
        raise ValueError('each fine-tuned transformer requires its own backbone instance')
    path = paths[0]
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('detector_architecture') != ARCHITECTURE
            or saved['backbone_revision'] != backbone.config._commit_hash):
        raise ValueError('incompatible transformer-only checkpoint/backbone')
    restore_encoder(backbone, saved['encoder'])
    backbone.requires_grad_(False).eval()
    config = backbone.config
    head = BareLinearHead(config.enc_hidden_d, 0, saved['height'], saved['width'],
        config.patch_height, config.patch_width, saved['dropout'], layers=0).to(device)
    head.load_state_dict(saved['head'], strict=True)
    return {path.stem:head.eval()}
