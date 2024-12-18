from dataclasses import dataclass
import logging

import torch
from torch import nn, Tensor

from .configs import VAEConfig, EncoderConfig, DecoderConfig
from .encoder import Encoder, Encoder3D, Encoder3DWavelet
from .decoder import Decoder, Decoder3D
from ..wavelet import dwt3d, idwt3d, HaarWavelet, Daubechies4Wavelet, ComplexDualTreeWavelet


@dataclass
class EncoderOutput:
    mean: Tensor
    logvar: Tensor
    
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
    value: Tensor
    
    def clone(self):
        return DecoderOutput(self.value.clone())
    
    def to(self, *args, **kwargs):
        return DecoderOutput(self.value.to(*args, **kwargs))


@dataclass
class VAEOutput:
    input: Tensor
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
    
    def encode(self, x: Tensor) -> EncoderOutput:
        z = self.encoder(x)
        z_mean, z_logvar = z.chunk(2, dim=1)
        return EncoderOutput(z_mean, z_logvar)
    
    def decode(self, z: Tensor) -> DecoderOutput:
        x = self.decoder(z)
        return DecoderOutput(x)
    
    def forward(
        self,
        x: Tensor,
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
    
    def encode(self, x: Tensor) -> EncoderOutput:
        z = self.encoder(x)
        z_mean, z_logvar = z.chunk(2, dim=-3)
        return EncoderOutput(z_mean, z_logvar)
    
    def decode(self, z: Tensor) -> DecoderOutput:
        x = self.decoder(z)
        return DecoderOutput(x)
    
    def forward(
        self,
        x: Tensor,
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


def _get_wavelet(config: EncoderConfig, max_level: int):
    wavelet_type = config.wavelet_type
    if wavelet_type is None:
        logging.warning('wavelet_type is None; use "haar" instead')
        wavelet_type = 'haar'
    
    param = config.wavelet_parameterized
    max_level = len(config.layer_out_dims) - 2
    
    if wavelet_type == 'haar':
        return HaarWavelet(param, max_level)
    
    if wavelet_type == 'daubechies4':
        return Daubechies4Wavelet(param, max_level)
    
    if wavelet_type == 'complexdualtree':
        return ComplexDualTreeWavelet(param, max_level)
    
    raise RuntimeError(f'unknown wavelet type: {wavelet_type}')


class VAE3DWavelet(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder3DWavelet(config.encoder)
        self.decoder = Decoder3D(config.decoder)
        self.wavelet = _get_wavelet(config.encoder, self.encoder.level)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_dwt(self, x: Tensor) -> list[Tensor]:
        # x := (b, f, c, h, w)
        # gray = 0.299R + 0.587G + 0.114B
        gray = x[:, :, 0, :, :] * 0.299 + x[:, :, 1, :, :] * 0.587 + x[:, :, 2, :, :] * 0.114
        # (b, f, h, w)
        dwt = dwt3d(gray, self.wavelet, level=self.encoder.level)
        
        cur = dwt
        ret = []
        while cur is not None:
            t = torch.stack([
                cur[key] for key
                # (b, f, h, w)
                in ('LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH')
            ], dim=1)
            # (b, c, f, h, w)
            ret.append(t)
            cur = cur.get('next', None)
        
        assert len(ret) == self.encoder.level
        
        return ret
    
    def encode(self, x: Tensor, dwt: list[Tensor]|None = None) -> EncoderOutput:
        if dwt is None:
            dwt = self.get_dwt(x)
        z = self.encoder(x, dwt)
        z_mean, z_logvar = z.chunk(2, dim=-3)
        return EncoderOutput(z_mean, z_logvar)
    
    def decode(self, z: Tensor) -> DecoderOutput:
        x = self.decoder(z)
        return DecoderOutput(x)
    
    def forward(
        self,
        x: Tensor,
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
