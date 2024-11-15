from typing import Sequence
import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    def __init__(
        self,
        shape: Sequence[int],
        total_size: int = 1024,
        dtype: torch.dtype = torch.float32,
        seed: int = 42
    ):
        super().__init__()
        self.shape = shape
        self.total_size = total_size
        self.dtype = dtype
        self.rng = torch.Generator().manual_seed(seed)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index: int):
        return torch.randn(self.shape, generator=self.rng, dtype=self.dtype)
