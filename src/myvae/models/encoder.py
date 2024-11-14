import functools

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .configs import EncoderConfig
from .down import EncoderBlock
from .mid import MidBlock


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        self.down_blocks = nn.ModuleList()
        for out_dim in config.layer_out_dims[:-1]:
            block = EncoderBlock(last_dim, out_dim, down=True, config=config)
            self.down_blocks.append(block)
            last_dim = out_dim
        block = EncoderBlock(last_dim, config.layer_out_dims[-1], down=False, config=config)
        self.down_blocks.append(block)
        last_dim = config.layer_out_dims[-1]
        
        self.mid_blocks = nn.ModuleList([
            MidBlock(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        self.norm_out = nn.GroupNorm(config.num_groups, config.layer_out_dims[-1], eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(config.layer_out_dims[-1], config.out_dim, kernel_size=3, padding=1)
    
    @property
    def _down_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.down_blocks
            ]
        else:
            return self.down_blocks
    
    @property
    def _mid_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.mid_blocks
            ]
        else:
            return self.mid_blocks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        
        z = self.conv_in(z)
        
        for block in self._down_blocks:
            z = block(z)
        
        for block in self._mid_blocks:
            z = block(z)
        
        z = self.norm_out(z)
        z = self.act_out(z)
        z = self.conv_out(z)
        
        return z
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        return self
