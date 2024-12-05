import torch
import torch.nn as nn
import torch.nn.functional as tf

from .configs import DecoderConfig
from .common import ResBlock, ResBlock3D


class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, up: bool, config: DecoderConfig):
        super().__init__()
        self.resblocks = nn.ModuleList()
        
        inout_dims = [(in_dim, out_dim)] + [(out_dim, out_dim)] * (config.num_resblocks_per_layer - 1)
        
        for in_d, out_d in inout_dims:
            block = ResBlock(in_d, out_d, config)
            self.resblocks.append(block)
        
        if up:
            self.up = Upsample2xNearest()
        else:
            self.up = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        x = self.up(x)
        return x


class DecoderBlock3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, up: bool, up_t: bool, config: DecoderConfig):
        super().__init__()
        self.resblocks = nn.ModuleList()
        
        inout_dims = [(in_dim, out_dim)] + [(out_dim, out_dim)] * (config.num_resblocks_per_layer - 1)
        
        for in_d, out_d in inout_dims:
            block = ResBlock3D(in_d, out_d, config)
            self.resblocks.append(block)
        
        if up or up_t:
            self.up = Upsample2xNearest()
            
            scale_factor = []
            
            if up_t:
                scale_factor.append(2.0)
            else:
                scale_factor.append(1.0)
            
            if up:
                scale_factor.append(2.0)
                scale_factor.append(2.0)
            else:
                scale_factor.append(1.0)
                scale_factor.append(1.0)
            
            self.scale_factor = [tuple(scale_factor)]
        else:
            self.up = nn.Identity()
            self.scale_factor = ()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        x = self.up(x, *self.scale_factor)
        return x


INTERPOLATION_MODE = [
    'nearest',
    'linear',
    'bilinear',
    'bicubic',
    'trilinear',
    'area',
    'nearest-exact',
]

class Upsample2xNearest(nn.Module):
    def forward(self, x: torch.Tensor, scale_factor: float|tuple[float,...] = 2.0) -> torch.Tensor:
        return tf.interpolate(x, scale_factor=scale_factor, mode='nearest')
