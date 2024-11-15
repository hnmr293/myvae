from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
import torch

import myvae


@dataclass(kw_only=True)  # non-default argument follows default argument
class TrainConfParams:
    n_epochs: int = 1000
    batch_size: int = 32
    grad_acc_steps: int = 1
    use_gradient_checkpointing: bool = False
    compile: bool = False
    optimizer: Callable[[Iterable], torch.optim.Optimizer] #= field(default_factory=partial(torch.optim.AdamW, lr=1e-4))
    scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] #= field(default_factory=partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0, total_iters=0))


@dataclass #(frozen=True)  # cannot inferit frozon dataclass from a non-frozen one
class TrainConf(TrainConfParams):
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


def load_config_from_arg():
    return load_config_from_path(parse_config_path())


def load_config_from_path(path: str|Path):
    return parse_config(path)


def parse_config_path():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('CONFIG', type=Path)
    path = p.parse_args().CONFIG
    return path


def parse_config(path: Path):
    from jsonargparse import ArgumentParser
    p = ArgumentParser()
    p.add_dataclass_arguments(TrainConfParams, 'train')
    p.add_argument('--project', type=str)
    p.add_argument('--dataset', type=torch.utils.data.Dataset)
    p.add_argument('--val_dataset', type=torch.utils.data.Dataset)
    p.add_argument('--dataloader', type=Callable[[torch.utils.data.Dataset, int], torch.utils.data.DataLoader])
    p.add_class_arguments(myvae.VAE, 'model')
    
    cfg = p.parse_path(path)
    init = p.instantiate_classes(cfg)
    
    dataloader_class = init.dataloader
    init.dataloader = dataloader_class(init.dataset, init.train.batch_size)
    init.val_dataloader = dataloader_class(init.val_dataset, 1)
    init.train.optimizer = init.train.optimizer(init.model.parameters())
    init.train.scheduler = init.train.scheduler(init.train.optimizer)
    
    init.train = TrainConf(**vars(init.train))
    
    return init
