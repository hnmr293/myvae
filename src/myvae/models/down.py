import torch
import torch.nn as nn
import torch.nn.functional as tf

from .configs import EncoderConfig
from .common import ResBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, down: bool, config: EncoderConfig):
        super().__init__()
        self.resblocks = nn.ModuleList()
        
        inout_dims = [(in_dim, out_dim)] + [(out_dim, out_dim)] * (config.num_resblocks_per_layer - 1)
        
        for in_d, out_d in inout_dims:
            block = ResBlock(in_d, out_d, config)
            self.resblocks.append(block)
        
        if down:
            self.down = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2),
            )
        else:
            self.down = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        x = self.down(x)
        return x
