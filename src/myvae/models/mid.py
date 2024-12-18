from torch import nn, Tensor

from .configs import EncoderConfig, DecoderConfig
from .common import ResBlock, ResBlock3D, Attention, FactorizedAttention3D


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
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.resblock(x)
        for block in self.attns:
            x = block(x)
        return x


class MidBlock3D(nn.Module):
    def __init__(self, dim: int, config: EncoderConfig | DecoderConfig):
        super().__init__()
        self.resblock = ResBlock3D(dim, dim, config)
        self.attns = nn.ModuleList([
            nn.Sequential(*[
                FactorizedAttention3D(dim, config),
                ResBlock3D(dim, dim, config),
            ])
            for _ in range(config.num_mid_attns)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.resblock(x)
        for block in self.attns:
            x = block(x)
        return x
