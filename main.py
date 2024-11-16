from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tvf
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import tqdm, gather_object
import wandb

from myvae import VAE, VAEOutput
from myvae.train import TrainConf
import myvae.train.loss as losses
from myvae.train.metrics import psnr, ssim


def train(
    acc: Accelerator,
    model: VAE,
    data: DataLoader,
    val_data: DataLoader,
    train_conf: TrainConf,
    hparam_config: dict,  # for model saving
):
    global_steps = 0
    
    #scaler = torch.GradScaler()  # Accelerator has its own GradScaler
    optimizer = train_conf.optimizer
    scheduler = train_conf.scheduler
    
    torch.set_float32_matmul_precision(train_conf.matmul_precision)
    model, data, val_data, optimizer, scheduler = acc.prepare(model, data, val_data, optimizer, scheduler)
    
    log_freq = train_conf.log_freq
    save_epochs = train_conf.save_every_n_epochs
    
    loss_fn = losses.Compose()
    for loss_dict in (train_conf.loss or []):
        loss_name, loss_weight = loss_dict['type'], loss_dict['weight']
        loss_fn.add(loss_name, loss_weight)
    
    
    for epoch in range(train_conf.n_epochs):
        with tqdm(data) as pbar:
            pbar.set_description(f'[Epoch {epoch}]')
            
            for step, batch in enumerate(pbar):
                with acc.autocast(), acc.accumulate(model):
                    x = batch
                    y: VAEOutput = model(x)
                    loss = loss_fn(y)
                    acc.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                pbar.set_postfix(loss=loss.item())
                
                if 0 < log_freq and (global_steps + 1) % log_freq == 0:
                    acc.log({
                        'train/loss': loss.item(),
                    }, step=global_steps)
                
                global_steps += 1
            
        # epoch end
        
        # validation
        val_results: list[VAEOutput] = []
        
        with torch.inference_mode():
            # for torch.compile
            torch.compiler.cudagraph_mark_step_begin()
            
            for batch in val_data:
                x = batch
                with acc.autocast():
                    y: VAEOutput = model(x)
                val_results.append(y)
        
            if acc.is_main_process:
                val_results = gather_object(val_results)
                
                def gather(fn: Callable[[VAEOutput], torch.Tensor]):
                    return torch.stack([fn(out) for out in val_result])
                
                val_loss = torch.mean(gather(lambda x: loss_fn(x)))
                val_kld_loss = torch.mean(gather(lambda x: losses.kld(x)))
                val_z_mean = torch.mean(gather(lambda x: x.encoder_output.mean))
                val_z_var = torch.mean(gather(lambda x: x.encoder_output.logvar)).exp()
                
                # compute validation loss
                acc.log({
                    'val/loss': val_loss.item(),
                    'val/KLD': val_kld_loss.item(),
                    'val/z_mean': val_z_mean.item(),
                    'val/z_var': val_z_var.item(),
                }, step=global_steps-1)
                
                # create images
                input_images = []
                generated_images = []
                for val_result in val_results:
                    inp_image = (val_result.input * 0.5 + 0.5).clamp(0, 1)
                    input_images.extend(inp_image)
                    gen_image = (val_result.decoder_output.value * 0.5 + 0.5).clamp(0, 1)
                    generated_images.extend(gen_image)
                input_images = torch.stack(input_images)
                generated_images = torch.stack(generated_images)
                
                nrow = (len(input_images) ** 0.5).__ceil__()
                image_left = make_grid(input_images, nrow=nrow)
                image_right = make_grid(generated_images, nrow=nrow)
                image = tvf.to_pil_image(torch.cat([image_left, image_right], dim=-1))
                
                acc.get_tracker('wandb').log({
                    'val/image': [wandb.Image(image)],
                }, step=global_steps-1)
                
                # compute metrics
                val_psnr = psnr(input_images, generated_images)
                val_ssim = ssim(tvf.rgb_to_grayscale(input_images), tvf.rgb_to_grayscale(generated_images), reduction='none')
                val_ssim_hist = np.histogram(val_ssim.reshape(-1).cpu().float(), bins=256, range=(-1, 1), density=True)
                acc.get_tracker('wandb').log({
                    'val/psnr': torch.mean(val_psnr).item(),
                    'val/ssim (grayscale)': wandb.Histogram(np_histogram=val_ssim_hist),
                }, step=global_steps-1)
                
                
                if 0 < save_epochs and (epoch + 1) % save_epochs == 0:
                    name = f'{epoch:05d}_{global_steps-1:08d}.ckpt'
                    dir = acc.get_tracker('wandb').run.id
                    save_model(
                        f'{dir}/{name}',
                        hparam_config,
                        acc.unwrap_model(model),
                        acc.unwrap_model(optimizer),
                        acc.unwrap_model(scheduler),
                        train_conf.hf_repo_id,
                    )

        acc.wait_for_everyone()


def save_model(
    path: str|Path,
    config: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    hf_repo_id: str|None,
):
    model_sd = model.state_dict()
    opt_sd = optimizer.state_dict()
    sched_sd = scheduler.state_dict()
    
    sd = {
        'state_dict': model_sd,
        'optimizer': opt_sd,
        'scheduler': sched_sd,
        'config': config,
    }
    
    if hf_repo_id is None:
        import os
        dir = Path(path).parent
        os.makedirs(dir, exist_ok=True)
        torch.save(sd, path)
    else:
        from io import BytesIO
        from huggingface_hub import HfApi
        io = BytesIO()
        torch.save(sd, io)
        io.seek(0)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=io,
            path_in_repo=path,
            repo_id=hf_repo_id,
        )


def load_model(path: str|Path, init):
    sd = torch.load(path, weights_only=True, map_location='cpu')
    
    metadata = sd.pop('config')
    
    from myvae.train import parse_dict
    
    conf = parse_dict(metadata)
    
    init.model.load_state_dict(sd['model'])
    init.optimizer.load_state_dict(sd['optimizer'])
    init.scheduler.load_state_dict(sd['scheduler'])
    
    #run_train(conf, metadata)
    return conf, metadata


def run_train(init, conf_dict):
    model = init.model
    data = init.dataloader
    val_data = init.val_dataloader
    train_conf = init.train
    
    assert isinstance(model, VAE)
    assert isinstance(data, DataLoader)
    assert isinstance(val_data, DataLoader)
    assert isinstance(train_conf, TrainConf)
    
    if train_conf.pretrained_weight is not None:
        load_model(train_conf.pretrained_weight, init)
    
    model.train()
    model.requires_grad_(True)
    
    if train_conf.use_gradient_checkpointing:
        model.apply_gradient_checkpointing()

    acc = Accelerator(
        log_with='wandb',
        gradient_accumulation_steps=train_conf.grad_acc_steps,
    )
    
    acc.init_trackers(
        project_name=init.project,
        config=conf_dict,
    )
    
    try:
        train(acc, model, data, val_data, train_conf, conf_dict)
    finally:
        acc.end_training()


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
