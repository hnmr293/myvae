from pathlib import Path

def load_model_for_inference(path: Path):
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
