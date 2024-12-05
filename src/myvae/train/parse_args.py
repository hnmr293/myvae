from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
import torch

import myvae


@dataclass(kw_only=True)  # non-default argument follows default argument
class TrainConfParams:
    n_epochs: int = 1000
    """学習エポック数"""
        
    batch_size: int = 32
    """バッチサイズ"""
    
    grad_acc_steps: int = 1
    """gradient accumulation を行うステップ数"""
    
    use_gradient_checkpointing: bool = False
    """gradient checkpointing (activation checkpointing) を行うか"""
    
    log_freq: int = 50
    """wandb でのログ頻度（ステップ単位）"""
    
    pretrained_weight: str|None = None
    """事前学習したモデルの重み"""
    
    optimizer: Callable[[Iterable], torch.optim.Optimizer] #= field(default_factory=partial(torch.optim.AdamW, lr=1e-4))
    """オプティマイザの設定"""
    
    scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] #= field(default_factory=partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0, total_iters=0))
    """学習率スケジューラの設定"""
    
    model_save_async: bool = False
    """モデルのアップロードを非同期で行うか"""
    
    save_every_n_epochs: int = 1
    """モデルの保存頻度（エポック単位）"""
    
    save_every_n_steps: int = 0
    """モデルの保存頻度（ステップ単位）"""
    
    validate_every_n_epochs: int = 1
    """バリデーションの頻度（エポック単位）"""
    
    validate_every_n_steps: int = 0
    """バリデーションの頻度（ステップ単位）"""
    
    hf_repo_id: str|None = None
    """モデルの保存先　未指定の場合ローカルに保存される"""
    
    loss: list[dict]|None = None
    """loss の指定
    
    {type: str, [weight: float = 1.0], [start_step: int = 0]} の形式"""
    
    val_loss: list[str]|None = None
    
    matmul_precision: str = 'highest'
    """torch.set_float32_matmul_precision に渡す引数"""


@dataclass #(frozen=True)  # cannot inferit frozon dataclass from a non-frozen one
class TrainConf(TrainConfParams):
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


def load_config_from_arg(only_model: bool = False):
    return load_config_from_path(parse_config_path(), only_model)


def load_config_from_path(path: str|Path, only_model: bool = False):
    return parse_config(path, only_model)


def parse_config_path():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('CONFIG', type=Path)
    path = p.parse_args().CONFIG
    return path


def parse_config(path: Path, only_model: bool = False):
    p = create_parser(only_model)
    cfg = p.parse_path(path)
    init = init_args(p, cfg, only_model)
    return init


def parse_dict(obj: dict, only_model: bool = False):
    p = create_parser(only_model)
    if only_model:
        obj = {k: obj[k] for k in ('project', 'train', 'model') if k in obj}
    cfg = p.parse_object(obj)
    init = init_args(p, cfg, only_model)
    return init


def create_parser(only_model: bool = False):
    from jsonargparse import ArgumentParser
    p = ArgumentParser()
    p.add_dataclass_arguments(TrainConfParams, 'train')
    p.add_argument('--project', type=str)
    if not only_model:
        p.add_argument('--dataset', type=torch.utils.data.Dataset)
        p.add_argument('--val_dataset', type=torch.utils.data.Dataset)
        p.add_argument('--dataloader', type=Callable[[torch.utils.data.Dataset, int], torch.utils.data.DataLoader])
        p.add_argument('--val_dataloader', type=Callable[[torch.utils.data.Dataset, int], torch.utils.data.DataLoader])
    p.add_argument('--model', type=torch.nn.Module)
    return p


def init_args(p, cfg, only_model: bool = False):
    init = p.instantiate_classes(cfg)
    
    if not only_model:
        init.dataloader = init.dataloader(init.dataset, init.train.batch_size)
        init.val_dataloader = init.val_dataloader(init.val_dataset, 1)  # 何かあると怖いのでバリデーション時のバッチサイズは1にしておく
        init.train.optimizer = init.train.optimizer(init.model.parameters())
        init.train.scheduler = init.train.scheduler(init.train.optimizer)
        
    init.train = TrainConf(**vars(init.train))
    
    return init
