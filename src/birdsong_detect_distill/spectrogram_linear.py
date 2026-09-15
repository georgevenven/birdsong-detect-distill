"""A framewise linear detector on normalized log-mel values, without an encoder."""
from types import SimpleNamespace

from torch import nn
from transformers import AutoConfig

ARCHITECTURE = 'raw_logmel_frame_linear_v1'


def frontend(model_id, revision):
    config = AutoConfig.from_pretrained(model_id, revision=revision,
        trust_remote_code=True, local_files_only=True)
    # Reuse only the audio normalization and frame dimensions, never encoder weights.
    config.enc_hidden_d, config.patch_height, config.patch_width = 128, 128, 1
    return SpectrogramTokens(config)


class SpectrogramTokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_values, valid_timebins=None):
        return SimpleNamespace(last_hidden_state=input_values.squeeze(1).transpose(1, 2))


class SpectrogramLinear(nn.Module):
    def __init__(self, dimension=128, hidden=0, height=1, width=1000,
                 patch_height=128, patch_width=1, dropout=0, layers=0):
        super().__init__()
        if (dimension, hidden, height, patch_height, patch_width, layers) != (128, 0, 1, 128, 1, 0):
            raise ValueError('expected one 128-bin frequency column per token')
        self.output = nn.Linear(128, 128)

    def forward(self, tokens, valid=None):
        return self.output(tokens).transpose(1, 2)
