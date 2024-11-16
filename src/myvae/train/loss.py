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
        self.loss_fns: list[tuple[LossFn, float]] = []
    
    def add(self, loss_fn: LossFn|str, weight: float = 1.0):
        if isinstance(loss_fn, str):
            klass = get_loss_fn(loss_fn)
            loss_fn = klass()
        
        self.loss_fns.append((loss_fn, weight))
    
    def __call__(self, out: VAEOutput) -> Tensor:
        if len(self.loss_fns) == 0:
            loss_fns = [(MSELoss(), 1.0)]
        else:
            loss_fns = self.loss_fns
        
        loss = None
        for loss_fn, weight in loss_fns:
            L = weight * loss_fn(out)
            if loss is None:
                loss = L
            else:
                loss = loss + L
        
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
        tf.binary_cross_entropy(pred, target)