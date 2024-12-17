from pathlib import Path
from concurrent.futures import Future
import gc
from typing import Callable, Any

import numpy as np
import torch
import torch.nn.functional as tf
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tvf
from torchvision.utils import make_grid
import einops
from accelerate import Accelerator
from accelerate.utils import tqdm, gather_object, TorchDynamoPlugin
import wandb

from myvae import VAE, VAE3D, VAE3DWavelet, VAEOutput
from myvae.train import TrainConf
import myvae.train.loss as losses
from myvae.train.metrics import psnr, ssim


class ModelSaver:
    def __init__(self, additional_params: dict[str, Any]):
        self.additional_params = additional_params
        self.last_saved_path: Path|None = None
    
    def save(self, path: str|Path, obj: dict[str, Any]) -> Future:
        path = Path(path)
        
        res = None
        if self.last_saved_path != path:
            res = self._save(path, obj | self.additional_params)
            self.last_saved_path = path
        
        if not isinstance(res, Future):
            f = Future()
            f.set_result(res)
            res = f
        
        return res
    
    def _save(self, path: Path, obj: dict[str, Any]):
        raise NotImplementedError()
    
    def close(self):
        pass


class ModelSaverLocal(ModelSaver):
    def __init__(self, acc: Accelerator, additional_params: dict[str, Any]):
        super().__init__(additional_params)
        self.acc = acc
    
    def _save(self, path: Path, obj: dict[str, Any]):
        import os
        dir = path.parent
        os.makedirs(dir, exist_ok=True)
        self.acc.save(obj, path)


class ModelSaverHf(ModelSaver):
    def __init__(self, hf_repo_id: str, use_async: bool, additional_params: dict[str, Any]):
        super().__init__(additional_params)
        
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.repo_id = hf_repo_id
        self.use_async = use_async
        self.last_future: Future|None = None
    
    def _save(self, path: Path, obj: dict[str, Any]):
        from io import BytesIO
        io = BytesIO()
        torch.save(obj, io)
        io.seek(0)
        
        upload_file = self.api.upload_file
        if self.use_async:
            from functools import partial
            upload_file = partial(self.api.run_as_future, upload_file)
        
        if self.last_future is not None:
            self.close()
        
        res = upload_file(
            path_or_fileobj=io,
            path_in_repo=str(path),
            repo_id=self.repo_id,
        )
        
        if isinstance(res, Future):
            self.last_future = res
    
    def close(self):
        if self.last_future is not None:
            # 前のアップロードが未完了だったら完了するまで待つ
            try:
                self.last_future.result()
            except Exception as e:
                import sys
                print(str(e), file=sys.stderr)
            self.last_future = None


def save_model(
    acc: Accelerator,
    saver: ModelSaver,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    steps: int,
):
    acc.wait_for_everyone()
    
    if acc.is_main_process:
        dir = acc.get_tracker('wandb').run.id
        name = f'{epoch:05d}_{steps:08d}.ckpt'
        path = f'{dir}/{name}'
        
        def replace(key: str):
            if isinstance(key, str):
                # torch.compile 対応
                key = key.replace('._orig_mod.', '.')
            return key
        
        def detach(t):
            if isinstance(t, dict):
                return {
                    replace(key): detach(val)
                    for key, val in t.items()
                }
            if isinstance(t, (list, tuple)):
                return type(t)(detach(v) for v in t)
            if isinstance(t, torch.Tensor):
                t = t.clone().detach().cpu()
            return t
        
        sd = {
            'state_dict': detach(acc.unwrap_model(model).state_dict()),
            'optimizer': detach(acc.unwrap_model(optimizer).state_dict()),
            'scheduler': detach(acc.unwrap_model(scheduler).state_dict()),
        }
        
        saver.save(path, sd)
    
    acc.wait_for_everyone()


def train(
    acc: Accelerator,
    model: VAE|VAE3D|VAE3DWavelet,
    data: DataLoader,
    val_data: DataLoader,
    train_conf: TrainConf,
    compile_options: dict[str, Any]|None,
    saver: ModelSaver,
):
    global_steps = 0
    
    #scaler = torch.GradScaler()  # Accelerator has its own GradScaler
    optimizer = train_conf.optimizer
    scheduler = train_conf.scheduler
    
    torch.set_float32_matmul_precision(train_conf.matmul_precision)
    
    if compile_options is not None:
        model = torch.compile(model, **compile_options)
    
    model, data, val_data, optimizer, scheduler = acc.prepare(model, data, val_data, optimizer, scheduler)
    
    log_freq = train_conf.log_freq
    save_freq = train_conf.save_every_n_steps
    save_epochs = train_conf.save_every_n_epochs
    val_freq = train_conf.validate_every_n_steps
    val_epochs = train_conf.validate_every_n_epochs
    
    loss_fn = losses.Compose()
    for loss_dict in (train_conf.loss or []):
        loss_name, loss_weight = loss_dict['type'], loss_dict['weight']
        loss_start_step = loss_dict.get('start_step', 0)
        loss_fn.add(loss_name, loss_weight, loss_start_step)
    
    acc.wait_for_everyone()
    
    for epoch in range(train_conf.n_epochs):
        with tqdm(data) as pbar:
            pbar.set_description(f'[Epoch {epoch}]')
            
            for step, batch in enumerate(pbar):
                lr = scheduler.get_last_lr()[0]  # for logging
                
                with acc.autocast(), acc.accumulate(model):
                    x = batch
                    y: VAEOutput = model(x)
                    loss = loss_fn(y)
                    acc.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    loss_fn.step()
                    optimizer.zero_grad()
                
                pbar.set_postfix(loss=loss.item())
                
                if 0 < log_freq and (global_steps + 1) % log_freq == 0:
                    acc.log({
                        'train/loss': loss.item(),
                        'lr': lr,
                    }, step=global_steps)
                
                if 0 < val_freq and (global_steps + 1) % val_freq == 0:
                    validate(acc, model, val_data, loss_fn, global_steps, train_conf)
                
                if 0 < save_freq and (global_steps + 1) % save_freq == 0:
                    save_model(acc, saver, model, optimizer, scheduler, epoch, global_steps)
                
                global_steps += 1
            
        # epoch end
        
        acc.log({
            'epoch': epoch,
        }, step=global_steps-1)
        
        # validation
        if 0 < val_epochs and (epoch + 1) % val_epochs == 0:
            validate(acc, model, val_data, loss_fn, global_steps-1, train_conf)
        
        # saving
        if 0 < save_epochs and (epoch + 1) % save_epochs == 0:
            save_model(acc, saver, model, optimizer, scheduler, epoch, global_steps-1)
        
        acc.wait_for_everyone()


def validate(
    acc: Accelerator,
    model: VAE|VAE3D|VAE3DWavelet,
    val_data: DataLoader,
    loss_fn: losses.Loss,
    global_steps: int,
    train_conf: TrainConf,
):
    acc.wait_for_everyone()
    
    val_results: list[VAEOutput] = []
    
    model_is_training = model.training
    model.eval()
    
    with torch.inference_mode():
        # for torch.compile
        torch.compiler.cudagraph_mark_step_begin()
        
        for batch in val_data:
            x = batch
            with acc.autocast():
                y: VAEOutput = model(x)
            val_results.append(y)
    
        val_results = gather_object(val_results)
        
        if acc.is_main_process:
            val_results = [out.to(device=acc.device) for out in val_results]
            
            def gather(fn: Callable[[VAEOutput], torch.Tensor], cat: bool = False):
                if cat:
                    return torch.cat([fn(out) for out in val_results])
                else:
                    return torch.stack([fn(out) for out in val_results])
            
            val_losses = [
                losses.get_loss_fn(loss_name)()
                for loss_name in (train_conf.val_loss or [])
            ]
            
            # compute validation loss
            val_total_loss = torch.mean(gather(lambda x: loss_fn(x, current_step=global_steps)))
            val_z_mean = torch.mean(gather(lambda x: x.encoder_output.mean.reshape(-1), cat=True))
            val_z_var = torch.mean(gather(lambda x: x.encoder_output.logvar.reshape(-1), cat=True)).exp()
            
            # z.mean の分散を計算する
            # 1. 郡内分散
            val_z_mean_var_intra = torch.mean(gather(lambda x: torch.var(x.encoder_output.mean)))
            # 2. 群間分散
            val_z_mean_var_inter = torch.var(gather(lambda x: torch.mean(x.encoder_output.mean))) * val_results[0].encoder_output.mean.numel()
            
            val_extra_losses = {
                f'val/loss/{fn.name.upper()}': torch.mean(gather(lambda x: fn(x))).item()
                for fn in val_losses
            }
            
            acc.log({
                'val/z/mean': val_z_mean.item(),
                'val/z/var': val_z_var.item(),
                'val/z/mean_within_var': val_z_mean_var_intra.item(),
                'val/z/mean_between_var': val_z_mean_var_inter.item(),
                'val/z/mean_var': (val_z_mean_var_intra + val_z_mean_var_inter).item(),
                'val/loss/total': val_total_loss.item(),
                **val_extra_losses,
            }, step=global_steps)
            
            # create images
            input_images = []
            generated_images = []
            for val_result in val_results:
                inp_image = (val_result.input * 0.5 + 0.5).clamp(0, 1)
                input_images.extend(inp_image)
                gen_image = (val_result.decoder_output.value * 0.5 + 0.5).clamp(0, 1)
                generated_images.extend(gen_image)
            
            width = min(t.size(-1) for t in input_images)
            height = min(t.size(-2) for t in input_images)
            
            if input_images[0].ndim == 3:
                # 2D VAE
                input_images = torch.stack([t[..., :height, :width] for t in input_images])
                generated_images = torch.stack([t[..., :height, :width] for t in generated_images])
                
                nrow = (len(input_images) ** 0.5).__ceil__()
                image_left = make_grid(input_images, nrow=nrow)
                image_right = make_grid(generated_images, nrow=nrow)
                image = tvf.to_pil_image(torch.cat([image_left, image_right], dim=-1))
                diff = (input_images - generated_images).abs()
                image_diff = tvf.to_pil_image(make_grid(diff, nrow=nrow))
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
                # (b f c h w) -> (b c h (f w))
                input_images = einops.rearrange(input_images, 'b f c h w -> b c h (f w)')
                generated_images = einops.rearrange(generated_images, 'b f c h w -> b c h (f w)')
                
                image_left = make_grid(input_images, nrow=1)
                image_right = make_grid(generated_images, nrow=1)
                image = tvf.to_pil_image(torch.cat([image_left, image_right], dim=-1))
                diff = (input_images - generated_images).abs()
                image_diff = tvf.to_pil_image(make_grid(diff, nrow=1))
            
            acc.get_tracker('wandb').log({
                'val/image': [wandb.Image(image)],
                'val/image/diff': [wandb.Image(image_diff)],
            }, step=global_steps)
            
            # compute metrics
            val_psnr = psnr(input_images, generated_images)
            val_ssim = ssim(tvf.rgb_to_grayscale(input_images), tvf.rgb_to_grayscale(generated_images), reduction='none')
            val_ssim_hist = np.histogram(val_ssim.reshape(-1).cpu().float(), bins=256, range=(-1, 1), density=True)
            acc.get_tracker('wandb').log({
                'val/psnr': torch.mean(val_psnr).item(),
                'val/ssim (grayscale)': wandb.Histogram(np_histogram=val_ssim_hist),
            }, step=global_steps)
    
    acc.wait_for_everyone()
    
    model.train(model_is_training)


def load_model(path: str|Path, init):
    sd = torch.load(path, weights_only=True, map_location='cpu')
    
    metadata = sd.pop('config')
    
    from myvae.train import parse_dict
    
    conf = parse_dict(metadata)
    
    sd_ = sd['state_dict']
    # remove compile wrapper
    sd_ = {k.replace('_orig_mod.', ''): v for k, v in sd_.items()}
    
    init.model.load_state_dict(sd_)
    init.train.optimizer.load_state_dict(sd['optimizer'])
    init.train.scheduler.load_state_dict(sd['scheduler'])
    
    return conf, metadata


def run_train(init, conf_dict):
    model = init.model
    data = init.dataloader
    val_data = init.val_dataloader
    train_conf = init.train
    
    assert isinstance(model, (VAE, VAE3D, VAE3DWavelet))
    assert isinstance(data, DataLoader)
    assert isinstance(val_data, DataLoader)
    assert isinstance(train_conf, TrainConf)
    
    if train_conf.pretrained_weight is not None:
        load_model(train_conf.pretrained_weight, init)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    model.train()
    model.requires_grad_(True)
    
    if train_conf.use_gradient_checkpointing:
        model.apply_gradient_checkpointing()

    acc = Accelerator(
        log_with='wandb',
        gradient_accumulation_steps=train_conf.grad_acc_steps,
        dynamo_backend='no',  # コンパイル範囲を自分で指定したいので no にしておく
    )
    
    # torch.compile 対応
    compile_options = TorchDynamoPlugin().to_kwargs()
    if len(compile_options) == 0 or str(compile_options.get('backend', 'no')).upper() == 'NO':
        # torch.compile 無効
        compile_options = None
    
    acc.init_trackers(
        project_name=init.project,
        config=conf_dict,
    )
    
    # エンコーダの縮小率
    r = 2 ** (len(model.encoder.config.layer_out_dims) - 1)
    
    # 画像サイズが異なるときの対応
    def collate_fn(data: list[torch.Tensor]):
        assert isinstance(data, (tuple, list))
        assert all(isinstance(t, torch.Tensor) for t in data)
        if data[0].ndim == 3:
            # t := (C, H, W)
            assert all(t.ndim == 3 for t in data)
            assert all(t.size(0) == 3 for t in data)
        else:
            # t := (F, C, H, W)
            assert all(t.ndim == 4 for t in data)
            assert all(t.size(1) == 3 for t in data)
        
        # 画質の観点から拡縮は行わず、バッチ内の一番小さい画像サイズに合わせる
        # 大きな画像は縮小するのではなく、一部を切り出す
        width = min(t.size(-1) for t in data)
        height = min(t.size(-2) for t in data)
        
        # エンコーダの縮小率に合わせる
        width = width & ~(r - 1)
        height = height & ~(r - 1)
        
        result = []
        for t in data:
            t = t[:, :height, :width]
            result.append(t)
        
        return torch.stack(result, dim=0)
    
    data.collate_fn = collate_fn
    val_data.collate_fn = collate_fn
    
    # モデルの保存先
    if train_conf.hf_repo_id is not None:
        saver = ModelSaverHf(
            train_conf.hf_repo_id,
            train_conf.model_save_async,
            additional_params={'config': conf_dict, 'compile': compile_options}
        )
    else:
        saver = ModelSaverLocal(
            acc,
            additional_params={'config': conf_dict, 'compile': compile_options}
        )
    
    if acc.is_main_process:
        import pprint
        print('=' * 80)
        print('  Training Setting')
        print('=' * 80)
        pprint.pprint(conf_dict)
    
    try:
        train(acc, model, data, val_data, train_conf, compile_options, saver)
    finally:
        acc.end_training()
        saver.close()


def main():
    import yaml
    from myvae.train import parse_config_path, parse_config
    
    conf_path = parse_config_path()
    
    # for logger
    with open(conf_path) as io:
        conf_yaml = yaml.safe_load(io)
    
    conf = parse_config(conf_path)
    
    run_train(conf, conf_yaml)


if __name__ == '__main__':
    main()
