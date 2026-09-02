import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn
from transformers import AutoModel


class DenseHead(nn.Module):
    def __init__(self, dimension, hidden, height, width, patch_height, patch_width, dropout=.1):
        super().__init__()
        self.height, self.width = height, width
        self.patch_height, self.patch_width = patch_height, patch_width
        self.project = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden))
        self.frequency = nn.Parameter(torch.zeros(1, height, 1, hidden))
        self.time = nn.Parameter(torch.zeros(1, 1, width, hidden))
        self.block = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, "gelu", batch_first=True, norm_first=True)
        self.output = nn.Linear(hidden, patch_height * patch_width)

    def forward(self, tokens, valid):
        batch = len(tokens)
        x = self.project(tokens).reshape(batch, self.height, self.width, -1) + self.frequency + self.time
        columns = torch.arange(self.width, device=x.device).repeat(self.height)
        padding = columns[None] >= ((valid.to(x.device) + self.patch_width - 1) // self.patch_width)[:, None]
        x = self.output(self.block(x.flatten(1, 2), src_key_padding_mask=padding)).reshape(
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


def load_backbone(model_id, device):
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    return model.requires_grad_(False).eval().to(device)


def load_detector(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    backbone = load_backbone(saved["backbone_id"], device)
    config = backbone.config
    head = DenseHead(config.enc_hidden_d, saved["hidden"], saved["height"], saved["width"],
        config.patch_height, config.patch_width, saved["dropout"]).to(device)
    head.load_state_dict(saved["head"])
    return backbone, head.eval(), saved


def loss(logits, targets, valid, smooth=True, tv_weight=1e-3):
    targets = F.avg_pool2d(targets[:, None], (31, 3), 1, (15, 1)).squeeze(1) if smooth else targets
    keep = torch.arange(logits.shape[-1], device=logits.device)[None, None] < valid[:, None, None].to(logits.device)
    bce = F.binary_cross_entropy_with_logits(logits[keep.expand_as(logits)], targets[keep.expand_as(logits)])
    probability = logits.sigmoid()
    tv = (probability[:, 1:] - probability[:, :-1]).abs().mean() + (probability[:, :, 1:] - probability[:, :, :-1]).abs().mean()
    return bce + tv_weight * tv
