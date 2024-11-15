from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as tf


# ref. https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
def psnr(x: Tensor, y: Tensor):
    # x := 0..1
    # y := 0..1
    
    mse = tf.mse_loss(x, y, reduction='none')
    mse = torch.mean(mse, dim=list(range(1, mse.ndim)))
    psnr = -10 * torch.log10(mse)
    return psnr


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def _get_gaussian_kernel2d(
    kernel_size: list[int], sigma: list[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype, device)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype, device)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

# ref. https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
def ssim(x: Tensor,
         y: Tensor,
         kernel_size: int|tuple[int,int] = 11,
         sigma: float|tuple[float,float]|None = 1.5,
         reduction: Literal['mean', 'L1', 'L2', 'none'] = 'mean',
         inner_dtype: torch.dtype = torch.float64,
         return_lcs: bool = False,
):
    # x := 0..1
    # y := 0..1
    
    K1 = 0.01
    K2 = 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # for raising
    x_shape = x.shape
    y_shape = y.shape
    
    x = x.to(inner_dtype)
    y = y.to(inner_dtype)
    
    x = x.view((-1, 1, x.size(-2), x.size(-1)))
    y = y.view((-1, 1, y.size(-2), y.size(-1)))
    
    if x.shape != y.shape:
        raise RuntimeError(f'unmatched shape: x={tuple(x_shape)}, y={tuple(y_shape)}')
    
    if not isinstance(kernel_size, (tuple, list)):
        kernel_size = (kernel_size, kernel_size)
    
    if sigma is None:
        sigma = -1
    if not isinstance(sigma, (tuple, list)):
        sigma = (sigma, sigma)
    
    # compute mu
    
    if 0 <= sigma[0] and 0 <= sigma[1]:
        kernel = _get_gaussian_kernel2d(kernel_size, sigma, inner_dtype, x.device)
        kernel = kernel[None, None, ...]
    else:
        kernel = torch.ones((1, 1, kernel_size[1], kernel_size[0]), dtype=inner_dtype, device=x.device) / (kernel_size[0] * kernel_size[1])

    # [left, right, top, bottom]
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    x_pad = tf.pad(x, padding, mode='reflect')
    y_pad = tf.pad(y, padding, mode='reflect')
    
    mu_x = tf.conv2d(x_pad, kernel)
    mu_y = tf.conv2d(y_pad, kernel)
    
    assert x.shape == mu_x.shape
    
    var_x = (x - mu_x) ** 2
    var_y = (y - mu_y) ** 2
    var_xy = (x - mu_x) * (y - mu_y)
    
    ssim_l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    ssim_cs = (2 * var_xy + C2) / (var_x + var_y + C2)
    ssim = ssim_l * ssim_cs
    
    match reduction:
        case 'mean':
            ssim = torch.mean(ssim)
        case 'L1':
            ssim = torch.mean(ssim.abs())
        case 'L2':
            ssim = torch.mean(ssim ** 2)
        case 'none':
            pass
        case _:
            raise RuntimeError(f'unknown reduction method: {reduction}')
    
    if return_lcs:
        return ssim, ssim_l, ssim_cs
    
    return ssim
