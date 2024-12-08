from itertools import product
import torch
from torch import Tensor
import torch.nn.functional as tf


class WaveletBase(torch.nn.Module):
    """基底となるウェーブレットクラス"""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def decomposition_filters(self) -> tuple[Tensor, Tensor]:
        """分解用フィルタを返す（低周波・高周波）"""
        raise NotImplementedError
        
    def reconstruction_filters(self) -> tuple[Tensor, Tensor]:
        """再構成用フィルタを返す（低周波・高周波）"""
        raise NotImplementedError


def dwt2d(image: Tensor, wavelet: WaveletBase, level: int = 1) -> dict[str, Tensor]:
    """
    2次元離散ウェーブレット変換を実行
    
    Parameters:
        image: 入力画像（2次元numpy配列）
        wavelet: ウェーブレットオブジェクト
        level: 分解レベル
        
    Returns:
        変換係数を含む辞書（'LL', 'LH', 'HL', 'HH'のキーを持つ）
    """
    if level < 1:
        raise ValueError(f'Level must be >= 1, but {level} is given.')
        
    # フィルタの取得
    lo_d, hi_d = wavelet.decomposition_filters()
    
    # x方向の変換
    L = _apply_filter_1d_down(image, lo_d, -1, pad='right')
    H = _apply_filter_1d_down(image, hi_d, -1, pad='right')
    
    # y方向の変換
    LL = _apply_filter_1d_down(L, lo_d, -2, pad='right')
    LH = _apply_filter_1d_down(L, hi_d, -2, pad='right')
    HL = _apply_filter_1d_down(H, lo_d, -2, pad='right')
    HH = _apply_filter_1d_down(H, hi_d, -2, pad='right')
    
    coeffs = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
    
    # 再帰的に分解
    if level > 1:
        sub_coeffs = dwt2d(LL, wavelet, level-1)
        coeffs['next'] = sub_coeffs
    
    return coeffs


def idwt2d(coeffs: dict[str, Tensor], wavelet: WaveletBase) -> Tensor:
    """
    2次元逆離散ウェーブレット変換
    
    Parameters:
        coeffs: dwt2dで得られた変換係数
        wavelet: ウェーブレットオブジェクト
        
    Returns:
        再構成された画像
    """
    # 再構成フィルタの取得
    lo_r, hi_r = wavelet.reconstruction_filters()
    
    def upsample(data: Tensor) -> Tensor:
        """アップサンプリング（ゼロ埋め）"""
        dims = (*data.shape[:-2], data.shape[-2]*2, data.shape[-1]*2)
        up = torch.zeros(dims, dtype=data.dtype, device=data.device)
        up[..., ::2, ::2] = data
        return up
    
    # 基本レベルの係数を取得して 0 埋めしつつ拡大
    data = torch.stack((coeffs['LL'], coeffs['LH'], coeffs['HL'], coeffs['HH']))
    up = upsample(data)
    LL, LH, HL, HH = up
    
    # 再構成
    LL = _apply_filter_2d(LL, lo_r, lo_r, pad='left')
    LH = _apply_filter_2d(LH, lo_r, hi_r, pad='left')
    HL = _apply_filter_2d(HL, hi_r, lo_r, pad='left')
    HH = _apply_filter_2d(HH, hi_r, hi_r, pad='left')
    
    reconstructed = LL + LH + HL + HH
    
    return reconstructed


def dwt3d(images: Tensor, wavelet: WaveletBase, level: int = 1) -> dict[str, Tensor]:
    """
    3次元離散ウェーブレット変換
    
    Parameters:
        images: 入力データ (3次元numpy配列)
        wavelet: ウェーブレットオブジェクト
        level: 分解レベル
        
    Returns:
        変換係数を含む辞書（'LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH'のキーを持つ）
    """
    if level < 1:
        raise ValueError(f'Level must be >= 1, but {level} is given.')
        
    if images.ndim < 3:
        raise ValueError(f'images.ndim must be 3 or greater, but {images.ndim} is given.')
    
    # フィルタの取得
    lo_d, hi_d = wavelet.decomposition_filters()
    
    # x方向の変換
    L, H = [
        _apply_filter_1d_down(images, d, -1, pad='right')
        for d in (lo_d, hi_d)
    ]
    
    # y方向の変換
    LL, LH, HL, HH = [
        _apply_filter_1d_down(image, d, -2, pad='right')
        for image, d in product((L, H), (lo_d, hi_d))
    ]
    
    # t方向の変換
    (
        LLL, HLL,
        LLH, HLH,
        LHL, HHL,
        LHH, HHH,
    ) = [
        _apply_filter_1d_down(image, d, -3, pad='right')
        for image, d in product((LL, LH, HL, HH), (lo_d, hi_d))
    ]
    
    coeffs = {
        'LLL': LLL, 'LLH': LLH, 'LHL': LHL, 'LHH': LHH,
        'HLL': HLL, 'HLH': HLH, 'HHL': HHL, 'HHH': HHH,
    }
    
    # 再帰的な分解
    if level > 1:
        sub_coeffs = dwt3d(LLL, wavelet, level-1)
        coeffs['next'] = sub_coeffs
    
    return coeffs


def idwt3d(coeffs: dict[str, Tensor], wavelet: WaveletBase) -> Tensor:
    """
    3次元逆離散ウェーブレット変換
    
    Parameters:
        coeffs: dwt3dで得られた変換係数
        
    Returns:
        再構成された3次元データ
    """
    # 再構成フィルタの取得
    lo_r, hi_r = wavelet.reconstruction_filters()
    
    # 再構成
    def apply(data: Tensor, filter: Tensor, dim: int):
        return _apply_filter_1d_up(data, filter, dim, pad='left')
    
    # t 方向
    LL = apply(coeffs['LLL'], lo_r, -3) + apply(coeffs['HLL'], hi_r, -3)
    LH = apply(coeffs['LLH'], lo_r, -3) + apply(coeffs['HLH'], hi_r, -3)
    HL = apply(coeffs['LHL'], lo_r, -3) + apply(coeffs['HHL'], hi_r, -3)
    HH = apply(coeffs['LHH'], lo_r, -3) + apply(coeffs['HHH'], hi_r, -3)
    
    # y 方向
    L = apply(LL, lo_r, -2) + apply(LH, hi_r, -2)
    H = apply(HL, lo_r, -2) + apply(HH, hi_r, -2)
    
    # z 方向
    data = apply(L, lo_r, -1) + apply(H, hi_r, -1)
    
    return data


def _upsample_1d(data: Tensor, dim: int) -> Tensor:
    """アップサンプリング（ゼロ埋め）"""
    dims = list(data.shape)
    dims[dim] *= 2
    up = torch.zeros(dims, dtype=data.dtype, device=data.device)
    assign_dims = [slice(None)] * up.ndim
    assign_dims[dim] = slice(None, None, 2)
    up[tuple(assign_dims)] = data
    return up


def _apply_filter_1d(data: Tensor, filter: Tensor, dim: int, pad: str = 'right') -> Tensor:
    """1次元フィルタを特定の軸に適用"""
    assert filter.ndim == 1
    
    pads = [(0, 0)] * data.ndim
    pads[-dim - 1] = (1, 0) if pad == 'left' else (0, 1)
    if 3 < len(pads):
        if not all(x == (0, 0) for x in pads[3:]):
            raise RuntimeError(f'reflective padding is only applicable for the axis -3..-1, but {dim} is given.')
        pads = pads[:3]
    pads = [x for xs in pads for x in xs]
    
    data = tf.pad(data, pads, mode='reflect')
    
    unfold = data.unfold(dim, size=filter.size(0), step=1)
    
    convolved = (unfold * filter).sum(dim=-1)
    
    return convolved


def _apply_filter_1d_down(data: Tensor, filter: Tensor, dim: int, pad: str) -> Tensor:
    """1次元フィルタを特定の軸に適用し偶数番めの要素だけをとってくる"""
    convolved = _apply_filter_1d(data, filter, dim, pad)
    
    indices = [slice(None)] * data.ndim
    indices[dim] = slice(None, None, 2)
    
    return convolved[tuple(indices)]


def _apply_filter_1d_up(data: Tensor, filter: Tensor, dim: int, pad: str) -> Tensor:
    """アップサンプリング＋1次元フィルタ"""
    data = _upsample_1d(data, dim)
    
    convolved = _apply_filter_1d(data, filter, dim, pad)
    
    return convolved


def _apply_filter_2d(data: Tensor, row_filter, col_filter, pad: str):
    data = _apply_filter_1d(data, row_filter, -1, pad)
    data = _apply_filter_1d(data, col_filter, -2, pad)
    return data
