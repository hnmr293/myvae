import numpy as np
import torch
from .base import WaveletBase


# 具体的なウェーブレットの実装

class WaveletFID(WaveletBase):
    """FIDフィルタを用いたウェーブレット"""
    pass


class WaveletFID1d(WaveletFID):
    """分離可能なFIDフィルタを用いたウェーブレット"""
    
    lo_d: torch.Tensor
    hi_d: torch.Tensor
    lo_r: torch.Tensor
    hi_r: torch.Tensor
    
    _name: str
    
    def __init__(self, parameterize: bool = False, max_level: int = 0):
        super().__init__(self._name)
        self.parameterized = parameterize
        self.max_level = max_level
        
        lo_d, hi_d, lo_r, hi_r = self.init_filters()
        
        self._init_kernel('lo_d', lo_d, parameterize, max_level)
        self._init_kernel('hi_d', hi_d, parameterize, max_level)
        self._init_kernel('lo_r', lo_r, False, 0)
        self._init_kernel('hi_r', hi_r, False, 0)
    
    def init_filters(self):
        raise NotImplementedError()
    
    def _init_kernel(
        self,
        name: str,
        kernel: torch.Tensor,
        parameterize: bool,
        max_level: int,
    ):
        assert kernel.ndim == 1
        
        if parameterize and 0 < max_level:
            kernel = torch.repeat_interleave(kernel[None], max_level, dim=0)
        if not hasattr(self, name):
            if parameterize:
                self.register_parameter(name, torch.nn.Parameter(torch.empty_like(kernel)))
            else:
                self.register_buffer(name, torch.empty_like(kernel))
        with torch.no_grad():
            buffer = getattr(self, name)
            buffer.copy_(kernel)

    def decomposition_filters(self, level: int = 0):
        if 0 < self.max_level:
            if self.max_level <= level:
                raise RuntimeError(f'level must be less than {self.max_level}')
            lo = self.lo_d[level]
            hi = self.hi_d[level]
        else:
            assert level == 0 or not self.parameterized
            lo = self.lo_d
            hi = self.hi_d
        return lo, hi
    
    def reconstruction_filters(self):
        return self.lo_r, self.hi_r
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # 旧バージョン対応
        lo_d = state_dict[f'{prefix}lo_d']
        hi_d = state_dict[f'{prefix}hi_d']
        if lo_d.ndim < self.lo_d.ndim:
            state_dict[f'{prefix}lo_d'] = torch.repeat_interleave(lo_d[None], self.lo_d.size(0), dim=0)
        if hi_d.ndim < self.hi_d.ndim:
            state_dict[f'{prefix}hi_d'] = torch.repeat_interleave(hi_d[None], self.hi_d.size(0), dim=0)
        #lo_r = state_dict[f'{prefix}lo_r']
        #hi_r = state_dict[f'{prefix}hi_r']
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class HaarWavelet(WaveletFID1d):
    """Haarウェーブレット"""
    _name = 'haar'
    def init_filters(self):
        lo_d = torch.tensor([1.0, 1.0]) / np.sqrt(2)
        hi_d = torch.tensor([1.0, -1.0]) / np.sqrt(2)
        lo_r = lo_d
        hi_r = -hi_d
        return lo_d, hi_d, lo_r, hi_r


class Daubechies4Wavelet(WaveletFID1d):
    """Daubechies-4ウェーブレット"""
    _name = 'db4'
    def init_filters(self):
        # Daubechies-4の係数
        h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
        h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
        h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        
        lo_d = torch.tensor([h0, h1, h2, h3])
        hi_d = torch.tensor([h3, -h2, h1, -h0])
        lo_r = torch.tensor([h3, h2, h1, h0])
        hi_r = torch.tensor([-h0, h1, -h2, h3])
        
        return lo_d, hi_d, lo_r, hi_r


class ComplexDualTreeWavelet(WaveletFID1d):
    """Dual-Tree Complex ウェーブレット"""
    _name = 'cdt'
    def init_filters(self):
        # 簡略化されたDual-Tree Complex Waveletのフィルタ
        # 実際の実装ではより精密なフィルタを使用する
        lo_d = torch.tensor([
            0.035226291882100656,
            -0.08544127388224149,
            -0.13501102001025458,
            0.4598775021193313,
            0.8068915093133388,
            0.3326705529500826
        ])
        hi_d = torch.tensor([
            -0.3326705529500826,
            0.8068915093133388,
            -0.4598775021193313,
            -0.13501102001025458,
            0.08544127388224149,
            0.035226291882100656
        ])
        lo_r = lo_d[::-1]
        hi_r = -hi_d[::-1]
        
        return lo_d, hi_d, lo_r, hi_r
