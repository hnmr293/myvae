import functools

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .configs import DecoderConfig
from .up import DecoderBlock, DecoderBlock3D
from .mid import MidBlock, MidBlock3D


class DecoderBase(nn.Module):
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device


class BottleneckDecoderBase(DecoderBase):
    """
    ボトルネック型VAEのデコーダの基本実装
    子クラスで
    - mid_blocks: nn.ModuleList
    - up_blocks: nn.ModuleList
    を定義すること
    """
    
    mid_blocks: nn.ModuleList
    up_blocks: nn.ModuleList
    
    @property
    def _up_blocks(self):
        """Gradient checkpointing を考慮してモジュールを取得する"""
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.up_blocks
            ]
        else:
            return self.up_blocks
    
    @property
    def _mid_blocks(self):
        """Gradient checkpointing を考慮してモジュールを取得する"""
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.mid_blocks
            ]
        else:
            return self.mid_blocks
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        return self


class Decoder(BottleneckDecoderBase):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        self.mid_blocks = nn.ModuleList([
            MidBlock(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        self.up_blocks = nn.ModuleList()
        for out_dim in config.layer_out_dims[:-1]:
            block = DecoderBlock(last_dim, out_dim, up=True, config=config)
            self.up_blocks.append(block)
            last_dim = out_dim
        block = DecoderBlock(last_dim, config.layer_out_dims[-1], up=False, config=config)
        self.up_blocks.append(block)
        
        self.norm_out = nn.GroupNorm(config.num_groups, config.layer_out_dims[-1], eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(config.layer_out_dims[-1], config.out_dim, kernel_size=3, padding=1)
    
    def forward(self, z: Tensor) -> Tensor:
        x = z
        
        x = self.conv_in(x)
        
        for block in self._mid_blocks:
            x = block(x)
        
        for block in self._up_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        
        return x


class Decoder3D(BottleneckDecoderBase):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        self.mid_blocks = nn.ModuleList([
            MidBlock3D(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        num_blocks = len(config.layer_out_dims)
        self.up_blocks = nn.ModuleList()
        for block_id, out_dim in enumerate(config.layer_out_dims):
            block = DecoderBlock3D(
                last_dim,
                out_dim,
                up=block_id < num_blocks-1,    # False if last block, otherwise True
                up_t=block_id < num_blocks-2,  # False if last two blocks, otherwise True
                config=config
            )
            self.up_blocks.append(block)
            last_dim = out_dim
        
        self.norm_out = nn.GroupNorm(config.num_groups, config.layer_out_dims[-1], eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(config.layer_out_dims[-1], config.out_dim, kernel_size=3, padding=1)
    
    def forward(self, z: Tensor) -> Tensor:
        x = z
        B, F, C, H, W = x.shape
        # (b, f, c, h, w)
        
        # conv2d は ndim=3 or 4 でないといけない
        x = x.view((-1, C, H, W))
        x = self.conv_in(x)
        x = x.view((B, F, -1, H, W))
        
        x = torch.movedim(x, 1, 2)
        # (b, c, f, h, w)
        
        for block in self._mid_blocks:
            x = block(x)
        
        for block in self._up_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = self.act_out(x)
        
        x = torch.movedim(x, 1, 2)
        # (b, f, c, h, w)
        
        # conv2d は ndim=3 or 4 でないといけない
        shape = x.shape
        x = x.reshape((-1, *shape[2:]))
        
        x = self.conv_out(x)
        
        x = x.view(*shape[:2], -1, *shape[3:])
        
        return x
