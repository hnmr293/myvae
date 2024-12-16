from typing import Callable, TypeAlias, Type

import torch
from torch import Tensor
import torch.nn.functional as tf

from myvae import VAEOutput, EncoderOutput


LossFn: TypeAlias = Callable[[VAEOutput], Tensor]


def get_loss_fn(name: str) -> 'Type[Loss]':
    import inspect
    from myvae.train import loss
    
    def pred(obj):
        return inspect.isclass(obj) and issubclass(obj, Loss)
    
    loss_fns = inspect.getmembers(loss, pred)
    for _, klass in loss_fns:
        if klass.name == name:
            return klass
    
    raise RuntimeError(f'unknown loss type: {name}')


def normalized_l1(x: Tensor, y: Tensor):
    assert x.shape == y.shape
    x = x.view((1, -1))
    y = y.view((1, -1))
    n = x.size(-1)
    return tf.l1_loss(x, y) / (n ** 0.5)


def kld(z: EncoderOutput) -> Tensor:
    mu = z.mean
    logvar = z.logvar
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


class Loss:
    # marker class
    
    name = '_base'
    
    def __call__(self, out: VAEOutput) -> Tensor:
        raise NotImplementedError()


class Compose(Loss):
    def __init__(self):
        self.loss_fns: list[tuple[LossFn, float, int]] = []
        self.steps = 0
    
    def step(self):
        self.steps += 1
    
    def add(self, loss_fn: LossFn|str, weight: float = 1.0, start_step: int = 0):
        if isinstance(loss_fn, str):
            klass = get_loss_fn(loss_fn)
            loss_fn = klass()
        
        self.loss_fns.append((loss_fn, weight, start_step))
    
    def __call__(self, out: VAEOutput, current_step: int|None = None) -> Tensor:
        if len(self.loss_fns) == 0:
            loss_fns = [(MSELoss(), 1.0, 0)]
        else:
            loss_fns = self.loss_fns
        
        if current_step is None:
            current_step = self.steps
        
        loss = None
        for loss_fn, weight, start_step in loss_fns:
            if current_step < start_step:
                continue
            
            L = weight * loss_fn(out)
            if loss is None:
                loss = L
            else:
                loss = loss + L
        
        assert loss is not None
        
        return loss


class L1Loss(Loss):
    name = 'l1'
    def __call__(self, out: VAEOutput) -> Tensor:
        return tf.l1_loss(out.decoder_output.value, out.input)


class MSELoss(Loss):
    name = 'mse'
    def __call__(self, out: VAEOutput) -> Tensor:
        return tf.mse_loss(out.decoder_output.value, out.input)


class KLDLoss(Loss):
    name = 'kld'
    def __call__(self, out: VAEOutput) -> Tensor:
        return kld(out.encoder_output)


class BCELoss(Loss):
    name = 'bce'
    def __call__(self, out: VAEOutput) -> Tensor:
        # VAE の出力が -1..1 だとして変換する
        pred = (out.decoder_output.value * 0.5 + 0.5).clamp(0, 1)
        target = (out.input * 0.5 + 0.5).clamp(0, 1)
        with torch.autocast(pred.device.type, enabled=False):
            return tf.binary_cross_entropy(pred.float(), target.float())


class GramMatrixL1Loss(Loss):
    name = 'gm'
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(self, out: VAEOutput):
        vec_target = out.input.flatten(-2).float()
        vec_pred = out.decoder_output.value.flatten(-2).float()
        with torch.autocast(vec_target.device.type, enabled=False):
            gm_target = vec_target @ vec_target.mT
            gm_pred = vec_pred @ vec_pred.mT
            loss = tf.l1_loss(gm_target, gm_pred)
            if self.normalize:
                n = vec_target.size(-1)
                loss = loss / (n ** 0.5)
        return loss


class LaplacianLoss(Loss):
    name = 'laplacian'
    def __init__(self, eight: bool = False):
        self.eight = eight
    
    @property
    def kernel4(self):
        return torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)[None]
    
    @property
    def kernel8(self):
        return torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)[None]
    
    @property
    def kernel(self):
        return self.kernel8 if self.eight else self.kernel4
    
    def apply_laplacian(self, x: Tensor, kernel: Tensor):
        input_dim = x.ndim
        if x.ndim == 2:
            x = x[None, None]
        elif x.ndim == 3:
            x = x[None]
        assert x.ndim == 4
        B, C, H, W = x.shape
        k = (
            kernel.view((1, 1, kernel.size(-2), kernel.size(-1))) # (1, 1, 3, 3)
            .repeat((C, 1, 1, 1)) # (C, 1, 3, 3)
            .to(device=x.device)
        )
        out = tf.conv2d(x.float(), k.float(), groups=C)
        return out.view((B, C, H-k.size(-2)+1, W-k.size(-1)+1)[-input_dim:])
    
    def __call__(self, out: VAEOutput):
        with torch.autocast(out.input.device.type, enabled=False):
            kernel = self.kernel.to(dtype=out.input.dtype, device=out.input.device)
            if out.input.ndim == 4:
                # 2D VAE
                pred = self.apply_laplacian(out.decoder_output.value, kernel)
                target = self.apply_laplacian(out.input, kernel)
            else:
                # 3D VAE
                assert out.input.ndim == 5
                pred = torch.stack([self.apply_laplacian(x, kernel) for x in out.decoder_output.value])
                target = torch.stack([self.apply_laplacian(x, kernel) for x in out.input])
            # L1 loss averaged over sequence
            loss = normalized_l1(pred, target)
        return loss


class GrayLaplacianLoss(LaplacianLoss):
    name = 'graylaplacian'
    def apply_laplacian(self, x, kernel):
        # gray = 0.299R + 0.587G + 0.114B
        if x.ndim == 3:
            assert x.size(0) == 3
            r, g, b = x  # (H, W)
            x = r * 0.299 + g * 0.587 + b * 0.114
            x = x[None]  # (1, H, W)
        elif x.ndim == 4:
            assert x.size(1) == 3
            r, g, b = x.unbind(dim=1)  # (B, H, W)
            x = r * 0.299 + g * 0.587 + b * 0.114
            x = x.unsqueeze(dim=1)     # (B, 1, H, W)
        return super().apply_laplacian(x, kernel)


class GMLaplacianLoss(LaplacianLoss):
    name = 'gmlaplacian'
    def __call__(self, out: VAEOutput):
        with torch.autocast(out.input.device.type, enabled=False):
            # ラプラシアンフィルタをかける
            kernel = self.kernel.to(dtype=out.input.dtype, device=out.input.device)
            if out.input.ndim == 4:
                # 2D VAE
                pred = self.apply_laplacian(out.decoder_output.value, kernel)
                target = self.apply_laplacian(out.input, kernel)
            else:
                # 3D VAE
                assert out.input.ndim == 5
                pred = torch.stack([self.apply_laplacian(x, kernel) for x in out.decoder_output.value])
                target = torch.stack([self.apply_laplacian(x, kernel) for x in out.input])
            # GM loss を計算する
            pred = pred.flatten(-2).float()
            target = target.flatten(-2).float()
            gm_target = pred @ pred.mT
            gm_pred = target @ target.mT
            loss = normalized_l1(gm_pred, gm_target)
        return loss


class GrayGMLaplacianLoass(GMLaplacianLoss):
    name = 'graygmlaplacian'
    def apply_laplacian(self, x, kernel):
        # gray = 0.299R + 0.587G + 0.114B
        if x.ndim == 3:
            assert x.size(0) == 3
            r, g, b = x  # (H, W)
            x = r * 0.299 + g * 0.587 + b * 0.114
            x = x[None]  # (1, H, W)
        elif x.ndim == 4:
            assert x.size(1) == 3
            r, g, b = x.unbind(dim=1)  # (B, H, W)
            x = r * 0.299 + g * 0.587 + b * 0.114
            x = x.unsqueeze(dim=1)     # (B, 1, H, W)
        return super().apply_laplacian(x, kernel)


class LpipsLoss(Loss):
    name = 'lpips'
    def __init__(self):
        from lpips import LPIPS
        self.lpips = LPIPS(net='vgg', verbose=False)
    def __call__(self, out: VAEOutput):
        lpips = self.lpips.to(out.input.device)
        
        # (b, c, h, w)
        # (b, f, c, h, w) -> (B, c, h, w)
        target = out.input.reshape((-1, *out.input.shape[-3:]))
        pred = out.decoder_output.value.reshape((-1, *out.decoder_output.value.shape[-3:]))
        
        dist = lpips(target, pred)
        
        return dist.mean()
