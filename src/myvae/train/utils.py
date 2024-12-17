import torch
from torch import nn, Tensor
import torch.nn.functional as tf
from torchvision.utils import make_grid as tv_make_grid
import einops

from myvae import VAE, VAE3D, VAE3DWavelet, VAEOutput


def restore_states(
    model: VAE|VAE3D|VAE3DWavelet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state_dict: dict[str, Tensor],
):
    model_sd = state_dict['state_dict']
    optim_sd = state_dict['optimizer']
    sched_sd = state_dict['scheduler']
    
    ret1 = restore_model_state(model, model_sd)
    ret2 = restore_optimizer_state(model, optimizer, optim_sd)
    ret3 = restore_scheduler_state(model, scheduler, sched_sd)
    
    return ret1, ret2, ret3


def restore_model_state(
    model: VAE|VAE3D|VAE3DWavelet,
    state_dict: dict[str, Tensor],
):
    # remove compile wrapper
    state_dict = {
        k.replace('_orig_mod.', ''): v
        for k, v in state_dict.items()
    }
    return model.load_state_dict(state_dict)


def restore_optimizer_state(
    model: VAE|VAE3D|VAE3DWavelet,
    optimizer: torch.optim.Optimizer,
    state_dict: dict,
):
    def is_old_wavelet():
        if not isinstance(model, VAE3DWavelet):
            return False
        model_requires_parameter = model.wavelet.parameterized
        
        required_params = len(list(model.parameters()))
        saved_params = len(state_dict['state'])
        required_wavelet_params = len(list(model.wavelet.parameters()))
        
        return (
            model_requires_parameter
            and required_wavelet_params != 0
            and saved_params + required_wavelet_params == required_params
        )
    
    if is_old_wavelet():
        # 旧バージョン対応
        # state_dict に保存された model.wavelet が buffer をもっていて、
        # 一方でこれから学習したい model が Parameter を要求するとき
        # state_dict に model.wavelet が必要とするパラメータがないので追加しておかないと読み込めない
        
        # optimizer
        #  + state: dict[int, dict[str, Tensor]]
        #            AdamW の場合 step, exp_avg, exp_avg_sq
        #  + param_groups
        #      - lr
        #        ...
        #        params: list[int]
        #      - lr
        #        ...
        #        params: list[int]
        #      - ...
        
        # state と param_groups[].params に要素が足りないので足す
        state = state_dict['state']
        all_params = [param_group['params'] for param_group in state_dict['param_groups']]
        
        num_saved_params = len(state_dict['state'])
        num_new_params = len(list(model.wavelet.parameters()))
        for param_index in range(num_saved_params, num_saved_params + num_new_params):
            assert param_index not in state
            state[param_index] = {}
            for params in all_params:
                assert param_index not in params
                params.append(param_index)
        
    return optimizer.load_state_dict(state_dict)


def restore_scheduler_state(
    model: VAE|VAE3D|VAE3DWavelet,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state_dict: dict,
):
    return scheduler.load_state_dict(state_dict)


def gather_images(outputs: list[VAEOutput]):
    """入力画像を表すテンソルと出力画像を表すテンソル、差の絶対値を表すテンソルを、全画像のサイズ（＋フレーム数）をクロップにより揃えたうえで返す"""
    input_images = []
    generated_images = []
    for output in outputs:
        inp_image = (output.input * 0.5 + 0.5).clamp(0, 1)
        input_images.extend(inp_image)
        gen_image = (output.decoder_output.value * 0.5 + 0.5).clamp(0, 1)
        generated_images.extend(gen_image)
    
    if len(input_images) == 0:
        return None, None, None
    
    width = min(t.size(-1) for t in input_images)
    height = min(t.size(-2) for t in input_images)
    
    if input_images[0].ndim == 3:
        # 2D VAE
        input_images = torch.stack([t[..., :height, :width] for t in input_images])
        generated_images = torch.stack([t[..., :height, :width] for t in generated_images])
        diff = (input_images - generated_images).abs()
    else:
        # 3D VAE
        assert input_images[0].ndim == 4
        frames = max(t.size(0) for t in input_images)
        input_images = torch.stack([
            t[:frames, :, :height, :width] if frames <= t.size(0)
            else tf.pad(t[:, :, :height, :width], (0, 0, 0, 0, 0, 0, 0, frames - t.size(0)), mode='constant', value=0)
            for t in input_images
        ])
        generated_images = torch.stack([
            t[:frames, :, :height, :width] if frames <= t.size(0)
            else tf.pad(t[:, :, :height, :width], (0, 0, 0, 0, 0, 0, 0, frames - t.size(0)), mode='constant', value=0)
            for t in generated_images
        ])
        diff = (input_images - generated_images).abs()
        ## (b f c h w) -> (b c h (f w))
        #input_images = einops.rearrange(input_images, 'b f c h w -> b c h (f w)')
        #generated_images = einops.rearrange(generated_images, 'b f c h w -> b c h (f w)')
    
    return input_images, generated_images, diff

def make_grid(images: Tensor):
    """画像をいい感じに横に並べる"""
    if images.ndim == 4:
        # 2D VAE
        # (b, c, h, w)
        nrow = (images.size(0) ** 0.5).__ceil__()
        images = tv_make_grid(images, nrow=nrow)
    else:
        # 3D VAE
        # (b, f, c, h, w)
        images = einops.rearrange(images, 'b f c h w -> b c h (f w)')
        images = tv_make_grid(images, nrow=1)
    
    return images
