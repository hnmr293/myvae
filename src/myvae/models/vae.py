from dataclasses import dataclass
from typing import overload

import torch
import torch.nn as nn
import torch.nn.functional as tf

from .configs import *
from .down import EncoderBlock
from .mid import MidBlock
from .up import DecoderBlock


@dataclass
class EncoderOutput:
    mean: torch.Tensor
    logvar: torch.Tensor
    
    @property
    def var(self):
        # exp({log s^2}) = s^2
        return torch.exp(self.logvar)
    
    @property
    def std(self):
        # exp({log (s^2)}/2) = {exp(log s^2)}^(1/2) = (s^2)^(1/2) = s
        return torch.exp(self.logvar / 2)
    
    def sample(self, rng: torch.Generator|None = None):
        a = self.mean
        e = torch.randn(a.shape, generator=rng, dtype=a.dtype, device=a.device)
        z = a + self.std * e
        # e ~ N(a,std^2)
        return z


@dataclass
class DecoderOutput:
    value: torch.Tensor


@dataclass
class VAEOutput:
    encoder_output: EncoderOutput
    decoder_output: DecoderOutput


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
    
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        z = self.encoder(x)
        z_mean, z_logvar = z.chunk(2, dim=1)
        return EncoderOutput(z_mean, z_logvar)
    
    def decode(self, z: torch.Tensor) -> DecoderOutput:
        x = self.decoder(z)
        return DecoderOutput(x)
    
    def forward(
        self,
        x: torch.Tensor,
        det: bool = False,
        rng: torch.Generator|None = None
    ) -> VAEOutput:
        encoded = self.encode(x)
        
        if det:
            z = encoded.mean
        else:
            z = encoded.sample(rng)
        
        y = self.decode(z)
        
        return VAEOutput(encoded, y)


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
