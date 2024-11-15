from PIL import Image
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
):
    global_steps = 0
    
    #scaler = torch.GradScaler()  # Accelerator has its own GradScaler
    optimizer = train_conf.optimizer
    scheduler = train_conf.scheduler
    
    model, data, val_data, optimizer, scheduler = acc.prepare(model, data, val_data, optimizer, scheduler)
    
    log_freq = train_conf.log_freq
    
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
            })
            
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
            #image.save(f'val.{epoch}.png')
            acc.get_tracker('wandb').log({
                'val/image': [wandb.Image(image)],
            }, step=global_steps-1)

        acc.wait_for_everyone()

def train_init(model: VAE, train_conf: TrainConf):
    model.train()
    model.requires_grad_(True)
    
    if train_conf.use_gradient_checkpointing:
        model.apply_gradient_checkpointing()


def main():
    import yaml
    from myvae.train import parse_config_path, parse_config
    
    conf_path = parse_config_path()
    
    # for logger
    with open(conf_path) as io:
        conf_yaml = yaml.load(io, Loader=yaml.SafeLoader)
    
    conf = parse_config(conf_path)
    
    model = conf.model
    data = conf.dataloader
    val_data = conf.val_dataloader
    train_conf = conf.train
    
    train_init(model, train_conf)
    
    acc = Accelerator(
        log_with='wandb',
        gradient_accumulation_steps=train_conf.grad_acc_steps,
    )
    
    acc.init_trackers(
        project_name=conf.project,
        config=conf_yaml,
    )
    
    try:
        train(acc, model, data, val_data, train_conf)
    finally:
        acc.end_training()


if __name__ == '__main__':
    main()
