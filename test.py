import torch
import myvae as vae

c = vae.VAEConfig()
v = vae.VAE(c)
v(torch.randn((1, 3, 512, 512)))
