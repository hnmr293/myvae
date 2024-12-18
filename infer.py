import sys
from pathlib import Path
from typing import Iterator

from PIL import Image


def parse_args():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('MODEL', type=Path)
    p.add_argument('IMAGE_OR_IMAGEDIR', type=Path)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--dtype', type=str, default='float32')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--crop', action='store_true')
    p.add_argument('--diff', action='store_true')
    p.add_argument('--max_frames', type=int, default=8)
    
    args = p.parse_args()
    
    return args


def load_model(path: Path):
    import torch
    sd = torch.load(path, weights_only=True, map_location='cpu')
    conf = sd.pop('config')
    
    from myvae import VAE, VAE3D, VAEWavelet, VAE3DWavelet
    from myvae.train import parse_dict
    
    if 'model' in conf and 'class_path' not in conf['model']:
        # 旧バージョン対応
        assert 'config' in conf['model']
        conf['model'] = {
            'class_path': 'myvae.VAE',
            'init_args': conf['model'],
        }
    init = parse_dict(conf, only_model=True, without_data=True)
    model = init.model
    assert isinstance(model, (VAE, VAE3D, VAEWavelet, VAE3DWavelet))
    
    sd = sd.pop('state_dict')
    # remove compile wrapper
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    
    model.load_state_dict(sd)
    
    model.eval().requires_grad_(False)
    
    return model


def iter_images(path: Path) -> Iterator[tuple[Path, Image.Image]]:
    assert path.exists()
    
    if path.is_dir():
        import glob, natsort
        files = natsort.natsorted([
            Path(p) for p in glob.glob(str(path / '*.*'))
            if p.endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
    else:
        files = [path]
    
    if len(files) == 0:
        raise RuntimeError(f'no files in {path}')
    
    for file in files:
        try:
            img = Image.open(file).convert('RGB')
        except:
            print(f'fail to load image: {file}', file=sys.stderr)
            continue
        yield file, img


def iter_videos(path: Path, max_frames: int) -> Iterator[tuple[Path, list[Image.Image]]]:
    assert path.exists()
    
    if not path.is_dir():
        raise RuntimeError(f'path must be directory: {path}')
    
    import glob, natsort
    dirs = natsort.natsorted([p for p in path.glob('*/') if p.is_dir()])
    
    for dir in dirs:
        imgs = sorted([
            p for p in dir.glob('*.*')
            if p.suffix in ('.jpg', '.jpeg', '.png', '.webp')
        ])
        imgs = [
            Image.open(file).convert('RGB')
            for file in imgs
        ]
        yield dir, imgs[:max_frames]


def crop(img: Image.Image, r: int):
    assert r & (r - 1) == 0, f'r must be power of 2, but {r} given'
    W, H = img.size
    w = W & ~(r-1)
    h = H & ~(r-1)
    x = (W - w) // 2
    y = (H - h) // 2
    cropped = img.crop((x, y, x+w, y+h))
    assert cropped.width == w
    assert cropped.height == h
    return cropped
    

def main():
    args = parse_args()
    
    file_path: Path = args.IMAGE_OR_IMAGEDIR
    
    if not file_path.exists():
        raise RuntimeError(f'{file_path} does not exist')
    
    import torch
    import torch.nn.functional as tf
    from torchvision.transforms.functional import to_tensor, normalize, to_pil_image
    import einops
    from myvae import VAE, VAE3D, VAEWavelet, VAE3DWavelet
    
    def calc_psnr(img1: torch.Tensor, img2: torch.Tensor):
        while img1.ndim < 4: img1 = img1[None]
        while img2.ndim < 4: img2 = img2[None]
        width = min(img1.size(-1), img2.size(-1))
        height = min(img1.size(-2), img2.size(-2))
        img1 = img1[..., :height, :width]
        img2 = img2[..., :height, :width]
        from myvae.train.metrics import psnr
        result = psnr(img1, img2)
        return result
    
    with torch.inference_mode():
        dtype = getattr(torch, args.dtype)
        device = torch.device(args.device)
        assert isinstance(dtype, torch.dtype)
        
        model = load_model(args.MODEL).to(dtype=dtype, device=device)
        r = 2 ** (len(model.encoder.config.layer_out_dims) - 1)  # 縮小率
        
        is_3d = isinstance(model, (VAE3D, VAE3DWavelet))
        
        if is_3d:
            iterator = iter_videos(args.IMAGE_OR_IMAGEDIR, args.max_frames)
        else:
            iterator = iter_images(args.IMAGE_OR_IMAGEDIR)
        
        for i, (filepath, img) in enumerate(iterator):
            if args.crop:
                if is_3d:
                    img = [crop(im, r) for im in img]
                else:
                    img = crop(img, r)
            
            if is_3d:
                t0 = torch.stack([to_tensor(im) for im in img])
            else:
                t0 = to_tensor(img)
            t = normalize(t0, [0.5], [0.5])
            t = t.to(dtype=dtype, device=device)
            
            rng = None if args.seed < 0 else torch.Generator(model.device).manual_seed(args.seed)
            out = model.forward(t[None], det=args.deterministic, rng=rng)  # add batch dim for t
            
            gen_t = out.decoder_output.value[0]
            gen_t = (gen_t * 0.5 + 0.5).clamp(0, 1)
            
            psnr = calc_psnr(t[None], out.decoder_output.value).item()
            
            if t.shape != gen_t.shape:
                t_pad_x = 0
                t_pad_y = 0
                gen_t_pad_x = t.size(-1) - gen_t.size(-1)
                gen_t_pad_y = t.size(-2) - gen_t.size(-2)
                
                if gen_t_pad_x < 0:
                    t_pad_x = -gen_t_pad_x
                    gen_t_pad_x = 0
                if gen_t_pad_y < 0:
                    t_pad_y = -gen_t_pad_y
                    gen_t_pad_y = 0
                
                t0 = tf.pad(t0, (0, t_pad_x, 0, t_pad_y))
                gen_t = tf.pad(gen_t, (0, gen_t_pad_x, 0, gen_t_pad_y))
            
            gen_t = gen_t.cpu()
            
            if is_3d:
                t0 = einops.rearrange(t0, 'f c h w -> c h (f w)')
                gen_t = einops.rearrange(gen_t, 'f c h w -> c h (f w)')
            
            cat_dim = -2 if is_3d else -1  # 2Dなら横、3Dなら縦に並べる
            
            if args.diff:
                diff = (t0 - gen_t).abs()
                img = torch.cat((t0, gen_t, diff), dim=cat_dim)
            else:
                img = torch.cat((t0, gen_t), dim=cat_dim)
            
            img = to_pil_image(img)
            img.save(f'{i:05d}.png')
            
            print(f'{filepath} PSNR = {psnr:.1f}')


if __name__ == '__main__':
    main()
