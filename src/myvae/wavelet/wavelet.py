import numpy as np
import torch
from .base import WaveletBase


# 具体的なウェーブレットの実装
class HaarWavelet(WaveletBase):
    """Haarウェーブレット"""
    def __init__(self):
        super().__init__('haar')
        self._init_filters()
    
    def _init_filters(self):
        def reset(name: str, kernel: torch.Tensor):
            if not hasattr(self, name):
                self.register_buffer(name, torch.empty_like(kernel))
            buffer = getattr(self, name)
            buffer.copy_(kernel)
        
        lo_d = torch.tensor([1.0, 1.0]) / np.sqrt(2)
        hi_d = torch.tensor([1.0, -1.0]) / np.sqrt(2)
        lo_r = lo_d
        hi_r = -hi_d
        
        reset('lo_d', lo_d)
        reset('hi_d', hi_d)
        reset('lo_r', lo_r)
        reset('hi_r', hi_r)
    
    def decomposition_filters(self):
        return self.lo_d, self.hi_d
    
    def reconstruction_filters(self):
        return self.lo_r, self.hi_r

class Daubechies4Wavelet(WaveletBase):
    """Daubechies-4ウェーブレット"""
    def __init__(self):
        super().__init__('db4')
        self._init_filters()
    
    def _init_filters(self):
        def reset(name: str, kernel: torch.Tensor):
            if not hasattr(self, name):
                self.register_buffer(name, torch.empty_like(kernel))
            buffer = getattr(self, name)
            buffer.copy_(kernel)
        
        # Daubechies-4の係数
        h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
        h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
        h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        
        lo_d = torch.tensor([h0, h1, h2, h3])
        hi_d = torch.tensor([h3, -h2, h1, -h0])
        lo_r = torch.tensor([h3, h2, h1, h0])
        hi_r = torch.tensor([-h0, h1, -h2, h3])
        
        reset('lo_d', lo_d)
        reset('hi_d', hi_d)
        reset('lo_r', lo_r)
        reset('hi_r', hi_r)
    
    def decomposition_filters(self):
        return self._lo_d, self._hi_d
    
    def reconstruction_filters(self):
        return self._lo_r, self._hi_r

class ComplexDualTreeWavelet(WaveletBase):
    """Dual-Tree Complex Wavelet"""
    def __init__(self):
        super().__init__('dtcwt')
        self._init_filters()
    
    def _init_filters(self):
        def reset(name: str, kernel: torch.Tensor):
            if not hasattr(self, name):
                self.register_buffer(name, torch.empty_like(kernel))
            buffer = getattr(self, name)
            buffer.copy_(kernel)
        
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
        
        reset('lo_d', lo_d)
        reset('hi_d', hi_d)
        reset('lo_r', lo_r)
        reset('hi_r', hi_r)
    
    def decomposition_filters(self):
        return self._lo_d, self._hi_d
    
    def reconstruction_filters(self):
        return self._lo_r, self._hi_r
