import functools

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import einops

from .configs import EncoderConfig
from .down import EncoderBlock, EncoderBlock3D
from .mid import MidBlock, MidBlock3D


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        self.down_blocks = nn.ModuleList()
        for out_dim in config.layer_out_dims[:-1]:
            block = EncoderBlock(last_dim, out_dim, down=True, config=config)
            self.down_blocks.append(block)
            last_dim = out_dim
        block = EncoderBlock(last_dim, config.layer_out_dims[-1], down=False, config=config)
        self.down_blocks.append(block)
        last_dim = config.layer_out_dims[-1]
        
        self.mid_blocks = nn.ModuleList([
            MidBlock(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        self.norm_out = nn.GroupNorm(config.num_groups, config.layer_out_dims[-1], eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(config.layer_out_dims[-1], config.out_dim, kernel_size=3, padding=1)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def _down_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.down_blocks
            ]
        else:
            return self.down_blocks
    
    @property
    def _mid_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.mid_blocks
            ]
        else:
            return self.mid_blocks
    
    def forward(self, x: Tensor) -> Tensor:
        z = x
        
        z = self.conv_in(z)
        
        for block in self._down_blocks:
            z = block(z)
        
        for block in self._mid_blocks:
            z = block(z)
        
        z = self.norm_out(z)
        z = self.act_out(z)
        z = self.conv_out(z)
        
        return z
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        return self


class Encoder3D(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        num_blocks = len(config.layer_out_dims)
        self.down_blocks = nn.ModuleList()
        for block_id, out_dim in enumerate(config.layer_out_dims):
            block = EncoderBlock3D(
                last_dim,
                out_dim,
                down=block_id < num_blocks-1,    # False if last block, otherwise True
                down_t=block_id < num_blocks-2,  # False if last two blocks, otherwise True
                config=config,
            )
            self.down_blocks.append(block)
            last_dim = out_dim
        
        self.mid_blocks = nn.ModuleList([
            MidBlock3D(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        self.norm_out = nn.GroupNorm(config.num_groups, config.layer_out_dims[-1], eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(config.layer_out_dims[-1], config.out_dim, kernel_size=3, padding=1)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def _down_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.down_blocks
            ]
        else:
            return self.down_blocks
    
    @property
    def _mid_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.mid_blocks
            ]
        else:
            return self.mid_blocks
    
    def forward(self, x: Tensor) -> Tensor:
        z = x
        B, F, C, H, W = z.shape
        # (b, f, c, h, w)
        
        # conv2d は ndim=3 or 4 でないといけない
        z = z.view((-1, C, H, W))
        z = self.conv_in(z)
        z = z.view((B, F, -1, H, W))
        
        z = torch.movedim(z, 1, 2)
        # (b, c, f, h, w)
        
        for block in self._down_blocks:
            z = block(z)
        
        for block in self._mid_blocks:
            z = block(z)
        
        z = self.norm_out(z)
        z = self.act_out(z)
        
        z = torch.movedim(z, 1, 2)
        # (b, f, c, h, w)
        
        # conv2d は ndim=3 or 4 でないといけない
        shape = z.shape
        z = z.reshape((-1, *shape[2:]))
        
        z = self.conv_out(z)
        
        z = z.view(*shape[:2], -1, *shape[3:])
        
        return z
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        return self


class EncoderWavelet(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        dwt_in_dim = 4  # (L,H)^2
        num_blocks = len(config.layer_out_dims)
        self.down_blocks = nn.ModuleList()
        self.dwt_blocks = nn.ModuleList()
        
        for block_id, out_dim in enumerate(config.layer_out_dims):
            if block_id != 0:
                # 最初の1層は元画像なので wavelet からの入力はなし
                dwt_block = WaveletBlock(dwt_in_dim, last_dim, config)
                self.dwt_blocks.append(dwt_block)
                last_dim *= 2
            
            down = block_id != num_blocks - 1
            
            block = EncoderBlock(
                last_dim,
                out_dim,
                down=down,
                config=config,
            )
            self.down_blocks.append(block)
            
            last_dim = out_dim
        
        self.mid_blocks = nn.ModuleList([
            MidBlock(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        self.norm_out = nn.GroupNorm(config.num_groups, last_dim, eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(last_dim, config.out_dim, kernel_size=3, padding=1)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def level(self):
        return len(self.down_blocks) - 1
    
    @property
    def _down_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.down_blocks
            ]
        else:
            return self.down_blocks
    
    @property
    def _dwt_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.dwt_blocks
            ]
        else:
            return self.dwt_blocks
    
    @property
    def _mid_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.mid_blocks
            ]
        else:
            return self.mid_blocks
    
    def forward(self, x: Tensor, dwts: list[Tensor]) -> Tensor:
        assert len(dwts) == self.level
        
        z = x
        
        z = self.conv_in(z)
        
        down_blocks = self._down_blocks
        dwt_blocks = self._dwt_blocks
        
        z = down_blocks[0](z)
        
        for dwt, dwt_block, block in zip(dwts, dwt_blocks, down_blocks[1:]):
            y = dwt_block(dwt)
            z = torch.cat((z, y), dim=1)
            z = block(z)
        
        for block in self._mid_blocks:
            z = block(z)
        
        z = self.norm_out(z)
        z = self.act_out(z)
        z = self.conv_out(z)
        
        return z
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        return self


class Encoder3DWavelet(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.config = config
        
        self.conv_in = nn.Conv2d(config.in_dim, config.layer_out_dims[0], kernel_size=3, padding=1)
        last_dim = self.conv_in.out_channels
        
        dwt_in_dim = 8  # (L,H)^3
        num_blocks = len(config.layer_out_dims)
        self.down_blocks = nn.ModuleList()
        self.dwt_blocks = nn.ModuleList()
        
        for block_id, out_dim in enumerate(config.layer_out_dims):
            if block_id != 0 and block_id != num_blocks - 1:
                # 最初の1層は元画像
                # 最後の1層は時間のダウンサンプルなし
                # なので wavelet からの入力はなし
                dwt_block = WaveletBlock3D(dwt_in_dim, last_dim, config)
                self.dwt_blocks.append(dwt_block)
                last_dim *= 2
            
            block = EncoderBlock3D(
                last_dim,
                out_dim,
                down=block_id < num_blocks-1,    # False if last block, otherwise True
                down_t=block_id < num_blocks-2,  # False if last two blocks, otherwise True
                config=config,
            )
            self.down_blocks.append(block)
            last_dim = out_dim
        
        self.mid_blocks = nn.ModuleList([
            MidBlock3D(last_dim, config)
            for _ in range(config.num_mid_layers)
        ])
        
        self.norm_out = nn.GroupNorm(config.num_groups, config.layer_out_dims[-1], eps=config.norm_eps)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(config.layer_out_dims[-1], config.out_dim, kernel_size=3, padding=1)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def level(self):
        return len(self.down_blocks) - 2
    
    @property
    def _down_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.down_blocks
            ]
        else:
            return self.down_blocks
    
    @property
    def _dwt_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.dwt_blocks
            ]
        else:
            return self.dwt_blocks
    
    @property
    def _mid_blocks(self):
        if self.training and getattr(self, 'gradient_checkpointing', False):
            return [
                functools.partial(checkpoint, mod, use_reentrant=False)
                for mod in self.mid_blocks
            ]
        else:
            return self.mid_blocks
    
    def forward(self, x: Tensor, dwts: list[Tensor]) -> Tensor:
        assert len(dwts) == self.level
        
        z = x
        B, F, C, H, W = z.shape
        # (b, f, c, h, w)
        
        # conv2d は ndim=3 or 4 でないといけない
        z = z.view((-1, C, H, W))
        z = self.conv_in(z)
        z = z.view((B, F, -1, H, W))
        
        z = torch.movedim(z, 1, 2)
        # (b, c, f, h, w)
        
        down_blocks = self._down_blocks
        dwt_blocks = self._dwt_blocks
        
        z = down_blocks[0](z)
        
        for dwt, dwt_block, block in zip(dwts, dwt_blocks, down_blocks[1:-1]):
            y = dwt_block(dwt)
            z = torch.cat((z, y), dim=1)
            z = block(z)
        
        z = down_blocks[-1](z)
        
        for block in self._mid_blocks:
            z = block(z)
        
        z = self.norm_out(z)
        z = self.act_out(z)
        
        z = torch.movedim(z, 1, 2)
        # (b, f, c, h, w)
        
        # conv2d は ndim=3 or 4 でないといけない
        shape = z.shape
        z = z.reshape((-1, *shape[2:]))
        
        z = self.conv_out(z)
        
        z = z.view(*shape[:2], -1, *shape[3:])
        
        return z
    
    def apply_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        return self


class WaveletBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: EncoderConfig):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.block = EncoderBlock(out_dim, out_dim, down=False, config=config)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: dwt (b, in_dim, h, w)
        
        # 1. in_dim -> out_dim
        x = self.conv(x)
        
        # 2. out_dim -> out_dim
        x = self.block(x)
        
        return x


class WaveletBlock3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: EncoderConfig):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.block = EncoderBlock3D(out_dim, out_dim, down=False, down_t=False, config=config)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: dwt (b, in_dim, f, h, w)
        B = x.size(0)
        
        # 1. in_dim -> out_dim
        # conv2d は ndim=3 or 4 でないといけない
        x = einops.rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.conv(x)
        x = einops.rearrange(x, '(b f) c h w -> b c f h w', b=B)
        
        # 2. out_dim -> out_dim
        x = self.block(x)
        
        return x
