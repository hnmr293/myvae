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
    
    args = p.parse_args()
    
    return args


def load_model(path: Path):
    import torch
    sd = torch.load(path, weights_only=True, map_location='cpu')
    conf = sd.pop('config')
    
    from myvae import VAE
    from myvae.train import parse_dict
    
    init = parse_dict(conf)
    model = init.model
    assert isinstance(model, VAE)
    
    sd = sd.pop('state_dict')
    # remove compile wrapper
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    
    model.load_state_dict(sd)
    
    model.eval().requires_grad_(False)
    
    return model


def iter_images(path: Path) -> Iterator[Image.Image]:
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
        yield img


def main():
    args = parse_args()
    
    file_path: Path = args.IMAGE_OR_IMAGEDIR
    
    if not file_path.exists():
        raise RuntimeError(f'{file_path} does not exist')
    
    import torch
    import torch.nn.functional as tf
    from torchvision.transforms.functional import to_tensor, normalize, to_pil_image
    
    with torch.inference_mode():
        dtype = getattr(torch, args.dtype)
        device = torch.device(args.device)
        assert isinstance(dtype, torch.dtype)
        
        model = load_model(args.MODEL).to(dtype=dtype, device=device)
        
        for i, img in enumerate(iter_images(args.IMAGE_OR_IMAGEDIR)):
            t0 = to_tensor(img)
            t = normalize(t0, [0.5], [0.5])
            t = t.to(dtype=dtype, device=device)
            
            rng = None if args.seed < 0 else torch.Generator(model.device).manual_seed(args.seed)
            out = model.forward(t[None], det=args.deterministic, rng=rng)  # add batch dim for t
            
            gen_t = out.decoder_output.value[0]
            gen_t = (gen_t * 0.5 + 0.5).clamp(0, 1)
            
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
            
            img = torch.cat((t0, gen_t.cpu()), dim=-1)
            img = to_pil_image(img)
            img.save(f'{i:05d}.png')


if __name__ == '__main__':
    main()
