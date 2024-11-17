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
        vec_target = out.input.flatten(2).float()
        vec_pred = out.decoder_output.value.flatten(2).float()
        with torch.autocast(vec_target.device.type, enabled=False):
            gm_target = vec_target @ vec_target.mT
            gm_pred = vec_pred @ vec_pred.mT
            loss = tf.l1_loss(gm_target, gm_pred)
            if self.normalize:
                n = vec_target.size(-1)
                loss = loss / (n ** 0.5)
        return loss
