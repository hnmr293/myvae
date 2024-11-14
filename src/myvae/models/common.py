import torch
import torch.nn as nn
import torch.nn.functional as tf
import einops

from .configs import EncoderConfig, DecoderConfig


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: EncoderConfig | DecoderConfig):
        super().__init__()
        self.norm1 = nn.GroupNorm(config.num_groups, in_dim, eps=config.norm_eps)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(config.num_groups, out_dim, eps=config.norm_eps)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        if in_dim == out_dim:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        x = self.conv0(x)
        return x + h


class Attention(nn.Module):
    def __init__(self, dim: int, config: EncoderConfig | DecoderConfig):
        super().__init__()
        self.norm = nn.GroupNorm(config.num_groups, dim, eps=config.norm_eps)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.num_heads = config.attention.num_heads
        self.attn_dropout = config.attention.dropout
        assert dim % self.num_heads == 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        # x := (B, C, H, W)
        B, C, H, W = x.shape
        
        h = x
        
        h = self.norm(h)
        h = einops.rearrange(h, 'b c h w -> b (h w) c')
        h = self.qkv(h)
        q, k, v = einops.rearrange(h, 'b n (k t d) -> t b k n d', t=3, k=self.num_heads)
        h = tf.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout)
        
        h = self.out(h)
        h = self.dropout(h)
        
        h = einops.rearrange(h, 'b k (h w) d -> b (k d) h w', h=H, w=W)
        #assert h.size(1) == C
        
        return x + h
