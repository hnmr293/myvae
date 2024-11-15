from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tvf
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import tqdm, gather_object
import wandb

from myvae import VAE, VAEOutput
from myvae.train import TrainConf


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
    
    model, data, val_data, optimizer, scheduler = acc.prepare(model, data, val_data, optimizer, scheduler)
    
    log_freq = train_conf.log_freq
    save_epochs = train_conf.save_every_n_epochs
    
    for epoch in range(train_conf.n_epochs):
        with tqdm(data) as pbar:
            pbar.set_description(f'[Epoch {epoch}]')
            
            for step, batch in enumerate(pbar):
                with acc.autocast(), acc.accumulate(model):
                    x = batch
                    out: VAEOutput = model(x)
                    y = out.decoder_output.value
                    loss = torch.nn.functional.mse_loss(y, x)
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
        val_losses: list[torch.Tensor] = []
        with torch.inference_mode(), acc.autocast():
            for batch in val_data:
                x = batch
                out: VAEOutput = model(x)
                y = out.decoder_output.value
                loss = torch.nn.functional.mse_loss(y, x)
                val_results.append(out)
                val_losses.append(loss)
        
        if acc.is_main_process:
            val_results = gather_object(val_results)
            val_loss = torch.mean(torch.cat([loss.reshape(-1) for loss in gather_object(val_losses)], dim=0))
            
            # compute validation loss
            acc.log({
                'val/loss': val_loss.item()
            }, step=global_steps-1)
            
            # create images
            input_images = []
            generated_images = []
            for val_result in val_results:
                inp_image = (val_result.input * 0.5 + 0.5).clamp(0, 1)
                input_images.extend(inp_image)
                gen_image = val_result.decoder_output.value.clamp(0, 1)
                generated_images.extend(gen_image)
            nrow = (len(input_images) ** 0.5).__ceil__()
            image_left = make_grid(input_images, nrow=nrow)
            image_right = make_grid(generated_images, nrow=nrow)
            image = tvf.to_pil_image(torch.cat([image_left, image_right], dim=-1))
            
            acc.get_tracker('wandb').log({
                'val/image': [wandb.Image(image)],
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
