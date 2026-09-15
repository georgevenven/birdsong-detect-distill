"""Restore every detection-path SongMAE weight, excluding the reconstruction decoder."""
import torch

from .pointwise_bare_linear import BareLinearHead

ARCHITECTURE = 'songmae_full_encoder_bare_linear_v1'
PARTS = ('songmae.patch_projection', 'songmae.patch_conv', 'songmae.encoder',
    'songmae.freq_embed', 'songmae.time_embed')


def encoder_key(name):
    return any(name == part or name.startswith(part + '.') for part in PARTS)


def encoder_state(backbone):
    return {k:v for k,v in backbone.state_dict().items() if encoder_key(k)}


def restore_encoder(backbone, state):
    expected = set(encoder_state(backbone))
    if set(state) != expected:
        raise ValueError('checkpoint does not contain the complete detection encoder')
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if unexpected or set(missing) != set(backbone.state_dict()) - expected:
        raise ValueError('unexpected encoder restore mismatch')


def load_heads(paths, backbone, device):
    if len(paths) != 1:
        raise ValueError('each fine-tuned encoder requires its own backbone instance')
    path = paths[0]
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('detector_architecture') != ARCHITECTURE
            or saved['backbone_revision'] != backbone.config._commit_hash):
        raise ValueError('incompatible full-encoder checkpoint/backbone')
    restore_encoder(backbone, saved['encoder'])
    backbone.requires_grad_(False).eval()
    config = backbone.config
    head = BareLinearHead(config.enc_hidden_d, 0, saved['height'], saved['width'],
        config.patch_height, config.patch_width, saved['dropout'], layers=0).to(device)
    head.load_state_dict(saved['head'], strict=True)
    return {path.stem:head.eval()}
