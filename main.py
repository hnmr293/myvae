import torch
from torch.utils.data import DataLoader
import tqdm

from myvae import VAE, VAEOutput
from myvae.train import TrainConf


def train(model: VAE, data: DataLoader, train_conf: TrainConf):
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
                    batch = batch.to(dtype=dtype, device=device)
                    x: VAEOutput = model(batch)
                    x = x.decoder_output.value
                    loss = torch.nn.functional.mse_loss(x, torch.zeros_like(x))
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
            
            pass


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
    train_conf = conf.train
    
    train_init(model, train_conf)
    
    train(model, data, train_conf)


if __name__ == '__main__':
    main()
