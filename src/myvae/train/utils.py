import torch
from torch import nn, Tensor

from myvae import VAE, VAE3D, VAE3DWavelet


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
