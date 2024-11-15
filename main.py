from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tvf
from torchvision.utils import make_grid
import tqdm

from myvae import VAE, VAEOutput
from myvae.train import TrainConf


def train(
    model: VAE,
    data: DataLoader,
    val_data: DataLoader,
    train_conf: TrainConf,
):
    global_steps = 0
    
    dtype = model.dtype
    device = model.device
    
    scaler = torch.GradScaler(device=device.type)
    optimizer = train_conf.optimizer
    scheduler = train_conf.scheduler
    
    for epoch in range(train_conf.n_epochs):
        with tqdm.tqdm(data) as pbar:
            pbar.set_description(f'[Epoch {epoch}]')
            
            for step, batch in enumerate(pbar):
                with torch.autocast(device_type=device.type):
                    x = batch.to(dtype=dtype, device=device)
                    out: VAEOutput = model(x)
                    y = out.decoder_output.value
                    loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
                    loss = loss / train_conf.grad_acc_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % train_conf.grad_acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                scheduler.step()
                
                pbar.set_postfix(loss=loss.item() * train_conf.grad_acc_steps)
                global_steps += 1
            
            # epoch end
            # validation
            val_resuls: list[VAEOutput] = []
            with torch.inference_mode(), torch.autocast(device_type=device.type):
                for batch in val_data:
                    batch = batch.to(dtype=dtype, device=device)
                    x: VAEOutput = model(batch)
                    val_resuls.append(x)
            # compute validation loss
            # create images
            val_images = []
            for val_result in val_resuls:
                val_image = val_result.decoder_output.value.clamp(0, 1)
                val_images.extend(val_image)
            nrow = (len(val_images) ** 0.5).__ceil__()
            image: Image.Image = tvf.to_pil_image(make_grid(val_images, nrow=nrow))
            image.save(f'val.{epoch}.png')


def train_init(model: VAE, train_conf: TrainConf):
    model.train()
    model.requires_grad_(True)
    
    if train_conf.use_gradient_checkpointing:
        model.apply_gradient_checkpointing()
    
    if train_conf.compile:
        torch.compile(model, fullgraph=True)
    
    model.cuda()
    
    # suppress warning
    setattr(train_conf.optimizer, '_opt_called', True)


def main():
    from myvae.train import load_config_from_arg
    
    conf = load_config_from_arg()
    
    model = conf.model
    data = conf.dataloader
    val_data = conf.val_dataloader
    train_conf = conf.train
    
    train_init(model, train_conf)
    
    train(model, data, val_data, train_conf)


if __name__ == '__main__':
    main()
