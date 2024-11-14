from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int = 1
    dropout: float = 0.0


@dataclass(frozen=True)
class EncoderConfig:
    in_dim: int = 3
    out_dim: int = 8
    layer_out_dims: list[int] = field(default_factory=lambda: [128, 256, 512, 512])
    num_resblocks_per_layer: int = 2
    num_groups: int = 32
    norm_eps: float = 1e-6
    num_mid_layers: int = 1
    num_mid_attns: int = 1
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    dropout: float = 0.0
    
    def __post_init__(self):
        if any(d % self.num_groups != 0 for d in self.layer_out_dims):
            raise RuntimeError(f'mismatched group_norm_dim')


@dataclass(frozen=True)
class DecoderConfig:
    in_dim: int = 4
    out_dim: int = 3
    layer_out_dims: list[int] = field(default_factory=lambda: [512, 512, 256, 128])
    num_resblocks_per_layer: int = 3
    num_groups: int = 32
    norm_eps: float = 1e-6
    num_mid_layers: int = 1
    num_mid_attns: int = 1
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    dropout: float = 0.0
    
    def __post_init__(self):
        if any(d % self.num_groups != 0 for d in self.layer_out_dims):
            raise RuntimeError(f'mismatched group_norm_dim')


@dataclass(frozen=True)
class VAEConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


