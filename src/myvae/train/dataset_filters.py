"""
ミニバッチに対してオーグメンテーションを行うクラス群
"""

import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

from torch import Tensor
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf


class Base(ABC):
    def __call__(self, x: Tensor) -> Tensor:
        self.expects(x)
        x = self.process(x)
        self.ensure(x)
        return x
    
    def expects(self, x: Tensor) -> None:
        pass
    
    def ensure(self, x: Tensor) -> None:
        pass
    
    @abstractmethod
    def process(self, x: Tensor) -> Tensor:
        pass


class Filters(list):
    def __call__(self, x: Tensor) -> Tensor:
        for filter in self:
            assert isinstance(filter, Base)
            x = filter(x)
        return x


# 機能的に、本来は dataclass にすべきではない
# jsonargparse から引数を渡したいが、
# いちいち __init__ を定義するのが面倒なのでこのようにする

@dataclass
class FlipLR(Base):
    """左右にフリップする"""
    p: float = 0.5
    def process(self, x):
        if random.random() < self.p:
            x = x.flip(dims=(-1,))
        return x

@dataclass
class FlipTB(Base):
    """上下にフリップする"""
    p: float = 0.5
    def expects(self, x):
        assert 2 <= x.ndim
    def process(self, x):
        if random.random() < self.p:
            x = x.flip(dims=(-2,))
        return x

@dataclass
class FlipFrames(Base):
    """時間をフリップする"""
    p: float = 0.5
    def expects(self, x):
        assert 4 <= x.ndim
    def process(self, x):
        if random.random() < self.p:
            # (..., f, c, h, w)
            x = x.flip(dims=(-4,))
        return x

@dataclass
class RandomCrop(Base):
    """指定されたサイズでランダムに切り出す"""
    size: tuple[int, int]
    def expects(self, x):
        assert 2 <= x.ndim
        h, w = x.shape[-2:]
        assert self.size[0] <= w
        assert self.size[1] <= h
    def ensure(self, x):
        assert x.size(-1) == self.size[0], f'{self.size}, {x.shape}'
        assert x.size(-2) == self.size[1], f'{self.size}, {x.shape}'
    def process(self, x):
        h, w = x.shape[-2:]
        pad_x_max = w - self.size[0]
        pad_y_max = h - self.size[1]
        pad_x = random.randint(0, pad_x_max)
        pad_y = random.randint(0, pad_y_max)
        return x[..., pad_y:pad_y+self.size[1], pad_x:pad_x+self.size[0]]

@dataclass
class ResizeBase(Base):
    def save_resize(self, x: Tensor, factor: float, mode: str) -> Tensor:
        # tf.interpolate needs batch dim
        orig_ndim = x.ndim
        if orig_ndim == 3:
            x = x[None]
        x = tf.interpolate(x, scale_factor=factor, mode=mode)
        if orig_ndim == 3:
            x = x[0]
        return x

@dataclass
class RandomResize(ResizeBase):
    """指定された範囲内でランダムにリサイズする"""
    min_width: int
    max_width: int
    mode: str = 'bicubic'
    def expects(self, x):
        assert 2 <= x.ndim
    def ensure(self, x):
        assert self.min_width <= x.size(-1) <= self.max_width
    def process(self, x):
        target_width = random.randint(self.min_width, self.max_width)
        factor = target_width / x.size(-1)
        return self.save_resize(x, factor, self.mode)

@dataclass
class RandomResize2(ResizeBase):
    """指定された範囲内でランダムにリサイズする"""
    widths: list[int]
    mode: str = 'bicubic'
    def expects(self, x):
        assert 2 <= x.ndim
    def process(self, x):
        target_width = random.choice(self.widths)
        factor = target_width / x.size(-1)
        return self.save_resize(x, factor, self.mode)
