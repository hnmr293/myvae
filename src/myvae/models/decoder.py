import functools

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .configs import DecoderConfig
from .up import DecoderBlock
from .mid import MidBlock


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        self.mid_blocks = nn.ModuleList([
            MidBlock(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        self.up_blocks = nn.ModuleList()
        for out_dim in config.layer_out_dims[:-1]:
            block = DecoderBlock(last_dim, out_dim, up=True, config=config)
            self.up_blocks.append(block)
            last_dim = out_dim
        block = DecoderBlock(last_dim, config.layer_out_dims[-1], up=False, config=config)
        self.up_blocks.append(block)
        
        self.norm_out = nn.GroupNorm(config.num_groups, config.layer_out_dims[-1], eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(config.layer_out_dims[-1], config.out_dim, kernel_size=3, padding=1)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def _up_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.up_blocks
            ]
        else:
            return self.up_blocks
    
    @property
    def _mid_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.mid_blocks
            ]
        else:
            return self.mid_blocks
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        
        x = self.conv_in(x)
        
        for block in self._mid_blocks:
            x = block(x)
        
        for block in self._up_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        
        return x
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        return self
