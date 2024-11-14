import torch
from torch.utils.data import DataLoader
#import torch.optim
import tqdm

import myvae as vae
from train.dataset import DummyDataset


N_EPOCHS = 100
N_DATA = 1024
BATCH_SIZE = 2
TOTAL_STEPS = N_DATA * N_EPOCHS // BATCH_SIZE
LR = 1e-4


def prepare():
    c = vae.VAEConfig()
    model = vae.VAE(c).train().cuda().requires_grad_(True)
    model.apply_gradient_checkpointing()

    data = DataLoader(
        DummyDataset((3, 512, 512), total_size=N_DATA),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    return model, data


def train(model: vae.VAE, data: DataLoader):
    global_steps = 0
    
    dtype = model.dtype
    device = model.device
    
    scaler = torch.GradScaler(device=device.type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    for epoch in range(N_EPOCHS):
        with tqdm.tqdm(data) as pbar:
            pbar.set_description(f'[Epoch {epoch}]')
            for step, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                with torch.autocast(device_type=device.type):
                    batch = batch.to(dtype=dtype, device=device)
                    x: vae.VAEOutput = model(batch)
                    x = x.decoder_output.value
                    loss = torch.nn.functional.mse_loss(x, torch.zeros_like(x))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.set_postfix(loss=loss.item())
                global_steps += 1


def main():
    model, data = prepare()
    train(model, data)


if __name__ == '__main__':
    main()
