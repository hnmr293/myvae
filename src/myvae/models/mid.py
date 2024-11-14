import torch
import torch.nn as nn
import torch.nn.functional as tf

from .configs import EncoderConfig, DecoderConfig
from .common import ResBlock, Attention


class MidBlock(nn.Module):
    def __init__(self, dim: int, config: EncoderConfig | DecoderConfig):
        super().__init__()
        self.resblock = ResBlock(dim, dim, config)
        self.attns = nn.ModuleList([
            nn.Sequential(*[
                Attention(dim, config),
                ResBlock(dim, dim, config),
            ])
            for _ in range(config.num_mid_attns)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resblock(x)
        for block in self.attns:
            x = block(x)
        return x
