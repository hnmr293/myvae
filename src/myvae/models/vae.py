import torch
import torch.nn as nn
import torch.nn.functional as tf

from .configs import *
from .down import EncoderBlock
from .mid import MidBlock
from .up import DecoderBlock


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
    
    def forward(
        self,
        x: torch.Tensor,
        det: bool = False,
        rng: torch.Generator|None = None
    ) -> torch.Tensor:
        z = self.encoder(x)
        
        z_mean, z_logvar = z.chunk(2, dim=1)
        z_std = torch.exp(z_logvar / 2)  # exp({log (s^2)}/2) = {exp(log s^2)}^(1/2) = (s^2)^(1/2) = s
        
        if det:
            z = z_mean
        else:
            e = torch.randn(z_mean.shape, generator=rng, dtype=z_mean.dtype, device=z_mean.device)
            # e ~ N(0,I)
            z = z_mean + z_std * e
        
        y = self.decoder(z)
        
        return y, z, z_mean, z_std


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        
        z = self.conv_in(z)
        
        for block in self.down_blocks:
            z = block(z)
        
        for block in self.mid_blocks:
            z = block(z)
        
        z = self.norm_out(z)
        z = self.act_out(z)
        z = self.conv_out(z)
        
        return z


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
