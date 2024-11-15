from typing import Callable, TypeAlias, Type

import torch
from torch import Tensor
import torch.nn.functional as tf

from myvae import VAEOutput


LossFn: TypeAlias = Callable[[VAEOutput, Tensor], Tensor]


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


class Loss:
    # marker class
    
    name = '_base'
    
    def __call__(self, pred: VAEOutput, target: Tensor) -> Tensor:
        raise NotImplementedError()


class Compose(Loss):
    def __init__(self):
        self.loss_fns: list[tuple[LossFn, float]] = []
    
    def add(self, loss_fn: LossFn|str, weight: float = 1.0):
        if isinstance(loss_fn, str):
            klass = get_loss_fn(loss_fn)
            loss_fn = klass()
        
        self.loss_fns.append((loss_fn, weight))
    
    def __call__(self, pred: VAEOutput, target: Tensor) -> Tensor:
        if len(self.loss_fns) == 0:
            loss_fns = [(MSELoss(), 1.0)]
        else:
            loss_fns = self.loss_fns
        
        loss = None
        for loss_fn, weight in loss_fns:
            L = weight * loss_fn(pred, target)
            if loss is None:
                loss = L
            else:
                loss = loss + L
        
        return loss


class L1Loss(Loss):
    name = 'l1'
    def __call__(self, pred: VAEOutput, target: Tensor) -> Tensor:
        return tf.l1_loss(pred, target)


class MSELoss(Loss):
    name = 'mse'
    def __call__(self, pred: VAEOutput, target: Tensor) -> Tensor:
        return tf.mse_loss(pred.decoder_output.value, target)


class KLDLoss(Loss):
    name = 'kld'
    def __call__(self, pred: VAEOutput, target: Tensor) -> Tensor:
        mu = pred.encoder_output.mean
        logvar = pred.encoder_output.logvar
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld
