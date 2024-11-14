from dataclasses import dataclass

import torch
import torch.nn as nn

from .configs import VAEConfig
from .encoder import Encoder
from .decoder import Decoder


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
        
        return VAEOutput(encoded, y)
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.encoder.apply_gradient_checkpointing(enabled)
        self.decoder.apply_gradient_checkpointing(enabled)
        return self
