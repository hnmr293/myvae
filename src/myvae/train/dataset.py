import glob
from pathlib import Path
import random
from typing import Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
from torchvision.io import VideoReader


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
        sort: bool = True,
        aug_flip_lr: bool = False,
        aug_flip_tb: bool = False,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        data = glob.glob(str(self.data_dir / '*.webp'))
        if sort:
            self.data = sorted(data)
        else:
            self.data = list(data)
        if 0 < (limit or 0):
            self.data = self.data[:limit]
        self.transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize([0.5], [0.5])
        ])
        self.aug_flip_lr = aug_flip_lr
        self.aug_flip_tb = aug_flip_tb
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        tensor = self.transform(Image.open(data).convert('RGB'))
        if self.aug_flip_lr and random.random() < 0.5:
            tensor = tensor.flip(dims=(-1,))
        if self.aug_flip_tb and random.random() < 0.5:
            tensor = tensor.flip(dims=(-2,))
        return tensor


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        limit: int|None = None,
        sort: bool = True,
        aug_flip_lr: bool = False,
        aug_flip_tb: bool = False,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        
        files = glob.glob(str(self.data_dir / '*.*'))
        data = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if sort:
            self.data = sorted(data)
        else:
            self.data = data
        
        if len(self.data) == 0:
            raise RuntimeError(f'empty data: {self.data_dir}')
        
        if 0 < (limit or 0):
            self.data = self.data[:limit]
        
        self.transform = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize([0.5], [0.5])
        ])
        
        self.aug_flip_lr = aug_flip_lr
        self.aug_flip_tb = aug_flip_tb
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        tensor = self.transform(Image.open(data).convert('RGB'))
        if self.aug_flip_lr and random.random() < 0.5:
            tensor = tensor.flip(dims=(-1,))
        if self.aug_flip_tb and random.random() < 0.5:
            tensor = tensor.flip(dims=(-2,))
        return tensor


class VideoDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        frames: int,
        fps: float|None = None,
        ss: float|None = None,
        limit: int|None = None,
        sort: bool = False,
        aug_flip_lr: bool = False,
        aug_flip_tb: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.frames = frames
        self.fps = fps
        self.ss = ss
        
        data = self.data_dir.rglob('*.mp4')
        if sort:
            self.data = sorted(data)
        else:
            self.data = list(data)
        if len(self.data) == 0:
            raise RuntimeError(f'empty data: {self.data_dir}')
        
        if 0 < (limit or 0):
            self.data = self.data[:limit]
        
        self.aug_flip_lr = aug_flip_lr
        self.aug_flip_tb = aug_flip_tb
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path = self.data[index]
        reader = VideoReader(str(path))
        
        meta = reader.get_metadata()['video']
        fps = meta['fps'][0]
        duration = meta['duration'][0]
        
        total_frames = (fps * duration).__floor__()
        
        # 適当に切り出す
        if total_frames < self.frames:
            raise RuntimeError(f'the number of frames = {total_frames}, but {self.frames} is required: {path}')
        
        # 何フレームに1回フレームを切り出すか
        frame_step = (
            1.0 if self.fps is None
            else fps / self.fps
        )
        frame_step = max(frame_step, 0.0)
        
        # 必要なフレーム数
        required_frames = (frame_step * self.frames).__ceil__()
        if total_frames < required_frames:
            raise RuntimeError(f'the number of frames = {total_frames}, but {required_frames} is required: {path}')
        
        ss = self.ss
        if ss is None or ss < 0:
            if total_frames == required_frames:
                ss = 0
            else:
                ss_frame = torch.randint(0, total_frames - required_frames, ()).item()
                ss = ss_frame / fps
        ss = max(ss, 0)
        
        reader.seek(ss, keyframes_only=True)  # pyav バックエンドは precise seek に対応していない
        
        frames = []
        current_frames = 0.0
        for _ in range(required_frames):
            try:
                frame = next(reader)
            except StopIteration:
                reader.seek(0, keyframes_only=True)
                frame = next(reader)
            if len(frames) == 0:
                # first one
                frames.append(frame['data'])
                continue
            current_frames += 1
            if frame_step <= current_frames:
                frames.append(frame['data'])
                current_frames -= frame_step
        
        assert len(frames) != 0
        
        while len(frames) < self.frames:
            # 何かが起きてフレーム数が足りないので、とりあえず繰り返しにしておく
            frames += frames
        
        frames = torch.stack(frames[:self.frames])
        
        frames = frames / 255
        frames = tvf.normalize(frames, [0.5], [0.5])
        
        if self.aug_flip_lr and random.random() < 0.5:
            frames = frames.flip(dims=(-1,))
        if self.aug_flip_tb and random.random() < 0.5:
            frames = frames.flip(dims=(-2,))
        
        return frames
