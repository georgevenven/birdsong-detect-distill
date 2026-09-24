import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn
from transformers import AutoModel


class DenseHead(nn.Module):
    def __init__(self, dimension, hidden, height, width, patch_height, patch_width, dropout=.1, layers=1):
        super().__init__()
        if layers < 0:
            raise ValueError("predictor layer count must be nonnegative")
        self.height, self.width = height, width
        self.patch_height, self.patch_width = patch_height, patch_width
        self.project = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden))
        self.frequency = nn.Parameter(torch.zeros(1, height, 1, hidden))
        self.time = nn.Parameter(torch.zeros(1, 1, width, hidden))
        self.block = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, "gelu", batch_first=True, norm_first=True)
        self.output = nn.Linear(hidden, patch_height * patch_width)
        # Preserve single-layer checkpoint keys and initialization order.
        self.extra_blocks = nn.ModuleList(nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout,
            "gelu", batch_first=True, norm_first=True) for _ in range(layers - 1))
        # Zero-layer ablation retains identical shared-weight initialization/RNG consumption.
        if layers == 0:
            self.block = None

    def forward(self, tokens, valid):
        batch = len(tokens)
        x = self.project(tokens).reshape(batch, self.height, self.width, -1) + self.frequency + self.time
        columns = torch.arange(self.width, device=x.device).repeat(self.height)
        padding = columns[None] >= ((valid.to(x.device) + self.patch_width - 1) // self.patch_width)[:, None]
        x = x.flatten(1, 2)
        if self.block is not None:
            x = self.block(x, src_key_padding_mask=padding)
        for block in self.extra_blocks:
            x = block(x, src_key_padding_mask=padding)
        x = self.output(x).reshape(
            batch, self.height, self.width, self.patch_height, self.patch_width)
        return x.permute(0, 1, 3, 2, 4).reshape(batch, self.height * self.patch_height, self.width * self.patch_width)


def clean_mask(probability, low=.22, high=.45, sigma=(2, 3), minimum_area=32, minimum_width=3):
    logits = np.log(np.clip(probability, 1e-5, 1 - 1e-5) / np.clip(1 - probability, 1e-5, 1))
    smooth = 1 / (1 + np.exp(-ndimage.gaussian_filter(logits, sigma)))
    mask = ndimage.binary_propagation(smooth >= high, mask=smooth >= low, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 5)))
    labels, count = ndimage.label(mask)
    for component in range(1, count + 1):
        y, x = np.where(labels == component)
        if len(x) < minimum_area or np.ptp(x) + 1 < minimum_width:
            mask[labels == component] = False
    return smooth, ndimage.binary_fill_holes(mask)


def load_backbone(model_id, device, revision=None):
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    return model.requires_grad_(False).eval().to(device)


def load_detector(path, device="cpu", revision=None):
    saved = torch.load(path, map_location=device, weights_only=True)
    backbone = load_backbone(saved["backbone_id"], device, revision or saved.get("backbone_revision"))
    config = backbone.config
    head = DenseHead(config.enc_hidden_d, saved["hidden"], saved["height"], saved["width"],
        config.patch_height, config.patch_width, saved["dropout"], saved.get("head_layers", 1)).to(device)
    head.load_state_dict(saved["head"])
    return backbone, head.eval(), saved


def supervision_mask(targets, valid):
    time = torch.arange(targets.shape[-1], device=targets.device)[None, None]
    return (time < valid[:, None, None].to(targets.device)) & (targets >= 0)


def loss(logits, targets, valid, smooth=True, tv_weight=1e-3, ignore_uncertain=False):
    if ignore_uncertain and (smooth or tv_weight != 0):
        raise ValueError("ignored pixels require hard-mask BCE without TV")
    targets = F.avg_pool2d(targets[:, None], (31, 3), 1, (15, 1)).squeeze(1) if smooth else targets
    keep = torch.arange(logits.shape[-1], device=logits.device)[None, None] < valid[:, None, None].to(logits.device)
    if ignore_uncertain:
        keep = supervision_mask(targets, valid)
        if not keep.any():
            return logits.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logits[keep.expand_as(logits)], targets[keep.expand_as(logits)])
    probability = logits.sigmoid()
    tv = (probability[:, 1:] - probability[:, :-1]).abs().mean() + (probability[:, :, 1:] - probability[:, :, :-1]).abs().mean()
    return bce + tv_weight * tv
