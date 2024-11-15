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
    
    for epoch in range(train_conf.n_epochs):
        with tqdm(data) as pbar:
            pbar.set_description(f'[Epoch {epoch}]')
            
            for step, batch in enumerate(pbar):
                with acc.autocast(), acc.accumulate(model):
                    x = batch
                    out: VAEOutput = model(x)
                    y = out.decoder_output.value
                    loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
                    acc.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                pbar.set_postfix(loss=loss.item())
                
                if (global_steps + 1) % 50 == 0:
                    acc.log({
                        'train/loss': loss.item(),
                    }, step=global_steps)
                
                global_steps += 1
            
        # epoch end
        
        # validation
        val_resuls: list[VAEOutput] = []
        with torch.inference_mode(), acc.autocast():
            for batch in val_data:
                x = batch
                y: VAEOutput = model(x)
                val_resuls.append(y)
        
        if acc.is_main_process:
            val_resuls = gather_object(val_resuls)
            # compute validation loss
            #acc.log({
            #    'val/loss': ...
            #})
            # create images
            val_images = []
            for val_result in val_resuls:
                val_image = val_result.decoder_output.value.clamp(0, 1)
                val_images.extend(val_image)
            nrow = (len(val_images) ** 0.5).__ceil__()
            image: Image.Image = tvf.to_pil_image(make_grid(val_images, nrow=nrow))
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
