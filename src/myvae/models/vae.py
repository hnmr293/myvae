from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from typing import Callable

import torch
from torch import nn, Tensor

from .configs import VAEConfig, EncoderConfig, DecoderConfig
from .encoder import Encoder, Encoder3D, EncoderWavelet, Encoder3DWavelet
from .decoder import Decoder, Decoder3D
from ..wavelet import dwt2d, dwt3d, WaveletFID1d, HaarWavelet, Daubechies4Wavelet, ComplexDualTreeWavelet


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



"""

普通の VAE

"""


class VAEBase(ABC, nn.Module):
    """VAEのベースクラス"""
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @abstractmethod
    def encode(self, x: Tensor) -> EncoderOutput:
        pass
    
    @abstractmethod
    def decode(self, z: Tensor) -> DecoderOutput:
        pass
    
    def forward(
        self,
        x: Tensor,
        det: bool = False,
        rng: torch.Generator|None = None,
    ) -> VAEOutput:
        encoded = self.encode(x)
        
        if det:
            z = encoded.mean
        else:
            z = encoded.sample(rng)
        
        y = self.decode(z)
        
        return VAEOutput(x, encoded, y)
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        for child in self.children():
            if hasattr(child, 'apply_gradient_checkpointing'):
                child.apply_gradient_checkpointing(enabled)
        return self


class VAEBase1(VAEBase):
    """
    普通の VAE のベースクラス
    
    以下の2つの子モジュールを持っていること
    - self.encoder
    - self.decoder
    
    また、self.encoder および self.decoder は Callable[[Tensor], Tensor] であること
    """
    
    # intersection type が欲しい……
    encoder: nn.Module | Callable[[Tensor], Tensor]
    decoder: nn.Module | Callable[[Tensor], Tensor]
    
    def encode(self, x: Tensor) -> EncoderOutput:
        z = self.encoder(x)
        z_mean, z_logvar = z.chunk(2, dim=-3)
        return EncoderOutput(z_mean, z_logvar)
    
    def decode(self, z: Tensor) -> DecoderOutput:
        x = self.decoder(z)
        return DecoderOutput(x)


class VAE(VAEBase1):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)


class VAE3D(VAEBase1):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder3D(config.encoder)
        self.decoder = Decoder3D(config.decoder)


"""

ウェーブレット変換を用いた VAE

"""


def _get_wavelet(config: EncoderConfig, max_level: int):
    """設定からウェーブレットを取得する便利メソッド"""
    
    wavelet_type = config.wavelet_type
    if wavelet_type is None:
        logging.warning('wavelet_type is None; use "haar" instead')
        wavelet_type = 'haar'
    
    param = config.wavelet_parameterized
    
    if wavelet_type == 'haar':
        return HaarWavelet(param, max_level)
    
    if wavelet_type == 'daubechies4':
        return Daubechies4Wavelet(param, max_level)
    
    if wavelet_type == 'complexdualtree':
        return ComplexDualTreeWavelet(param, max_level)
    
    raise RuntimeError(f'unknown wavelet type: {wavelet_type}')


def _gather_dwt(
    dwt: dict[str, Tensor],
    keys: tuple[str,...],
    stack_dim: int,
    next_key: str = 'next',
) -> list[Tensor]:
    """
    { ..., next: { ... } } 形式の辞書を再帰的に探索して、与えられた keys に対応する Tensor のリストを返す
    同じ階層に属する Tensor は torch.stack で結合される
    """

    cur = dwt
    ret = []
    while cur is not None:
        t = torch.stack([cur[key] for key in keys], dim=stack_dim)
        ret.append(t)
        cur = cur.get(next_key, None)
    
    return ret


class VAEWavelet1dBase(VAEBase):
    """
    分離可能なフィルタによるウェーブレット変換を用いた VAE のベースクラス
    
    以下の3つの子モジュールを持っていること
    - self.encoder
    - self.decoder
    - self.wavelet
    
    self.encoder および self.decoder は Callable[[Tensor, list[Tensor]], Tensor] であること
    self.wavelet は WaveletFID1d であること
    """
    
    # intersection type が欲しい……
    encoder: nn.Module | Callable[[Tensor, list[Tensor]], Tensor]
    decoder: nn.Module | Callable[[Tensor, list[Tensor]], Tensor]
    wavelet: WaveletFID1d
    
    @abstractmethod
    def dwt(self, gray: Tensor) -> list[Tensor]:
        """グレースケール化された画像を分解する"""
        pass
    
    def decompose(self, x: Tensor) -> list[Tensor]:
        """
        入力をウェーブレット変換により分解する
        入力は RGB 画像であることを前提にする
        """
        
        # x := (b, ..., c, h, w)
        
        assert x.size(-3) == 3, f'画像がRGB形式ではありません：shape={x.shape}'
        
        # グレースケール化
        # gray = 0.299R + 0.587G + 0.114B
        r, g, b = x.unbind(dim=-3)
        gray = r * 0.299 + g * 0.587 + b * 0.114
        # (b, ..., h, w)
        
        return self.dwt(gray)
    
    def encode(self, x: Tensor, dwt: list[Tensor]|None = None) -> EncoderOutput:
        if dwt is None:
            dwt = self.decompose(x)
        z = self.encoder(x, dwt)
        z_mean, z_logvar = z.chunk(2, dim=-3)
        return EncoderOutput(z_mean, z_logvar)
    
    def decode(self, z: Tensor) -> DecoderOutput:
        x = self.decoder(z)
        return DecoderOutput(x)


class VAEWavelet(VAEWavelet1dBase):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = EncoderWavelet(config.encoder)
        self.decoder = Decoder(config.decoder)
        self.wavelet = _get_wavelet(config.encoder, self.encoder.level)

    def dwt(self, gray: Tensor) -> list[Tensor]:
        # x := (b, h, w)
        
        dwt = dwt2d(gray, self.wavelet, level=self.wavelet.max_level)
        
        keys = ('LL', 'LH', 'HL', 'HH')
        ret = _gather_dwt(dwt, keys, stack_dim=1)
        # 各キーに対応するテンソルは (b, h, w) になっている
        # 返ってきたテンソルは (b, c, h, w) にする
        
        assert len(ret) == self.wavelet.max_level
        
        return ret


class VAE3DWavelet(VAEWavelet1dBase):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder3DWavelet(config.encoder)
        self.decoder = Decoder3D(config.decoder)
        self.wavelet = _get_wavelet(config.encoder, self.encoder.level)

    def dwt(self, gray: Tensor) -> list[Tensor]:
        # x := (b, f, h, w)
        
        dwt = dwt3d(gray, self.wavelet, level=self.wavelet.max_level)
        
        keys = ('LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH')
        ret = _gather_dwt(dwt, keys, stack_dim=1)
        # 各キーに対応するテンソルは (b, f, h, w) になっている
        # 返ってきたテンソルは (b, c, f, h, w) にする
        
        assert len(ret) == self.wavelet.max_level
        
        return ret
