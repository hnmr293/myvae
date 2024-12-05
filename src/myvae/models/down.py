import torch
import torch.nn as nn
import torch.nn.functional as tf

from .configs import EncoderConfig
from .common import ResBlock, ResBlock3D


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


class EncoderBlock3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, down: bool, down_t: bool, config: EncoderConfig):
        super().__init__()
        self.resblocks = nn.ModuleList()
        
        inout_dims = [(in_dim, out_dim)] + [(out_dim, out_dim)] * (config.num_resblocks_per_layer - 1)
        
        for in_d, out_d in inout_dims:
            block = ResBlock3D(in_d, out_d, config)
            self.resblocks.append(block)
        
        padding = {
            # stride=1 のとき (1, 1)
            # stride=2 のとき (0, 1)
            (False, False): (0, 0, 0, 0, 0, 0),
            (False, True): (1, 1, 1, 1, 0, 1),
            (True, False): (0, 1, 0, 1, 1, 1),
            (True, True): (0, 1, 0, 1, 0, 1),
        }[(down, down_t)]
        # ZeroPadNd の引数は (left, right, top, bottom, front, back) の形（ひどい）
        
        stride= {
            (False, False): (1, 1, 1),
            (False, True): (2, 1, 1),
            (True, False): (1, 2, 2),
            (True, True): (2, 2, 2),
        }[(down, down_t)]
        # Conv3d の stride は dim の順番
        
        if not down and not down_t:
            self.down = nn.Identity()
        else:
            self.down = nn.Sequential(
                nn.ZeroPad3d(padding),
                nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=stride),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        x = self.down(x)
        return x
