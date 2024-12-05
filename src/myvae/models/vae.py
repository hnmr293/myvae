from dataclasses import dataclass

import torch
import torch.nn as nn

from .configs import VAEConfig, EncoderConfig, DecoderConfig
from .encoder import Encoder, Encoder3D
from .decoder import Decoder, Decoder3D


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
        ## torch.randn_like は torch.Generator をとらない
        ## torch.randn() を使うと torch.compile がこけることがある（環境による？）
        ## torch.compile が呼べて Generator も使えるように empty_like().normal_() を使う
        ## 参考：
        ##   https://github.com/pytorch/pytorch/issues/27072
        ##   https://github.com/pytorch/pytorch/pull/136780
        #e = torch.empty_like(a).normal_(generator=rng)
        
        z = a + self.std * e
        # z ~ N(a,std^2)
        return z
    
    def clone(self):
        return EncoderOutput(self.mean.clone(), self.logvar.clone())
    
    def to(self, *args, **kwargs):
        return EncoderOutput(self.mean.to(*args, **kwargs), self.logvar.to(*args, **kwargs))


@dataclass
class DecoderOutput:
    value: torch.Tensor
    
    def clone(self):
        return DecoderOutput(self.value.clone())
    
    def to(self, *args, **kwargs):
        return DecoderOutput(self.value.to(*args, **kwargs))


@dataclass
class VAEOutput:
    input: torch.Tensor
    encoder_output: EncoderOutput
    decoder_output: DecoderOutput
    
    def clone(self):
        return VAEOutput(self.input.clone(), self.encoder_output.clone(), self.decoder_output.clone())
    
    def to(self, *args, **kwargs):
        return VAEOutput(self.input.to(*args, **kwargs), self.encoder_output.to(*args, **kwargs), self.decoder_output.to(*args, **kwargs))


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
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
        
        return VAEOutput(x, encoded, y)
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.encoder.apply_gradient_checkpointing(enabled)
        self.decoder.apply_gradient_checkpointing(enabled)
        return self


class VAE3D(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder3D(config.encoder)
        self.decoder = Decoder3D(config.decoder)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        z = self.encoder(x)
        z_mean, z_logvar = z.chunk(2, dim=-3)
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
        
        return VAEOutput(x, encoded, y)
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.encoder.apply_gradient_checkpointing(enabled)
        self.decoder.apply_gradient_checkpointing(enabled)
        return self
