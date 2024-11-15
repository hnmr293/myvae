from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt


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


class WebpDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        limit: int|None = None,
    ):
        super().__init__()
        import glob
        self.data_dir = Path(data_dir)
        self.data = glob.glob(str(self.data_dir / '*.webp'))
        if 0 < (limit or 0):
            self.data = self.data[:limit]
        self.transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        return self.transform(Image.open(data).convert('RGB'))
