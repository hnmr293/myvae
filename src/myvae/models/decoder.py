import torch
import torch.nn as nn

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
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        
        x = self.conv_in(x)
        
        for block in self.mid_blocks:
            x = block(x)
        
        for block in self.up_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        
        return x
