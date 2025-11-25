import torch
import torch.nn.functional as F
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL
from test_diffusionmodel import UNet, NoiseScheduler
from test_warpingnetwork import WarpingCloth
from torchvision import transforms
from PIL import Image

# Commented out IPython magic to ensure Python compatibility.
class VitonPipeline():

  def __init__(self, output_size, checkpoint_ddpm, checkpoint_warping, vae_autoencoder):

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.checkpoint_ddpm = checkpoint_ddpm
    self.checkpoint_warping = checkpoint_warping

    self.size = output_size

    self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    # Load pretrained Stable Diffusion VAE
    self.vae = vae_autoencoder
    self.vae.eval()  # keep frozen

    self.viton = UNet(9, 4).to(self.device)
    self.viton_opt = torch.optim.AdamW(self.viton.parameters(), lr=0.00001, betas=(0.5, 0.999))

    self.scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear", device=self.device)

  def encode_latents(self, images):
    # images: [B,3,H,W], range [-1,1]
    with torch.no_grad():
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # SD scaling factor
    return latents

  def decode_latents(self, latents):
    latents = latents / 0.18215
    with torch.no_grad():
        imgs = self.vae.decode(latents).sample
    return imgs

  def __call__(self, person_img, cloth_img):

    self.viton.load_state_dict(self.checkpoint_ddpm['model_state_dict'])
    self.viton_opt.load_state_dict(self.checkpoint_ddpm['optimizer_state_dict'])

    warping = WarpingCloth((256, 256), self.checkpoint_warping)

    self.input_viton = ((warping(person_img, cloth_img) - 0.5)/0.5).to(self.device)
    self.agnostic_mask = self.transform(warping.agnostic_mask).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)

    z_test_input_viton = self.encode_latents(self.input_viton)
    z_test_mask_image = F.interpolate(self.agnostic_mask, size=z_test_input_viton.shape[-2:], mode="nearest")

    timesteps_to_sample = torch.linspace(self.scheduler.timesteps - 1, 0, 200).to(self.device).long()
    x_current, _ = self.scheduler.get_noisy_image(z_test_input_viton, timesteps_to_sample[0])

    with torch.no_grad():
      for i, t in enumerate(timesteps_to_sample):
        t_prev = timesteps_to_sample[i + 1] if i < len(timesteps_to_sample) - 1 else 0

        input_to_unet = torch.cat([x_current, z_test_input_viton, z_test_mask_image[:, 0:1, :, :]], dim=1)

        predicted_noise = self.viton(input_to_unet, torch.tensor([t]).to(self.device))

        alpha_bar_t_val = self.scheduler.alphas_cumprod[t].item()
        x_0_pred = (x_current - math.sqrt(1.0 - alpha_bar_t_val) * predicted_noise) / math.sqrt(alpha_bar_t_val)

        #x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        alpha_bar_t_prev_val = self.scheduler.alphas_cumprod[t_prev].item()
        x_current = math.sqrt(alpha_bar_t_prev_val) * x_0_pred + math.sqrt(1.0 - alpha_bar_t_prev_val) * predicted_noise

        #x_current = torch.clamp(x_current, -1.0, 1.0)

        x_0_pred = self.decode_latents(x_0_pred)

    return x_0_pred



