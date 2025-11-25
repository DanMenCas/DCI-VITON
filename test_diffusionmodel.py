
from test_warpingnetwork import WarpingCloth

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from torchvision.transforms import ToPILImage
from torchvision import models
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL


def save_tensor_as_image(tensor, filename="output.png"):
    # Check tensor shape
    if tensor.shape != (1, 3, 512, 512):
        raise ValueError("Expected tensor shape [1, 3, 512, 512]")

    # Remove batch dimension
    tensor = tensor.squeeze(0)  # Shape: [3, 512, 512]

    # Convert tensor to PIL Image
    # Assuming tensor is on GPU, move to CPU and convert to numpy
    tensor = tensor.cpu().detach()

    # If tensor is in [0, 1], scale to [0, 255]
    tensor = ((tensor + 1)/2)*255

    # Convert to uint8
    tensor = tensor.clamp(0, 255).byte()

    # Permute dimensions from [C, H, W] to [H, W, C] for PIL
    tensor = tensor.permute(1, 2, 0)  # Shape: [512, 512, 3]

    # Convert to numpy array
    image_array = tensor.numpy()

    # Create PIL Image
    image = Image.fromarray(image_array)

    # Save the image
    image.save(filename)
    print(f"Image saved as {filename}")

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_schedule="linear", device='cpu'):
        self.timesteps = timesteps
        self.device = torch.device(device)

        if beta_schedule == "linear":
            betas = self._linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)

    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps, dtype=torch.float32)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        t = torch.arange(timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = f_t / f_t[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(betas, 0.0001, 0.999)

    def get_noisy_image(self, x_0, t, noise=None):
        """
        Adds noise to the original image x_0 at time t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        if t.ndim == 1:
            alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        else:
            alpha_bar_t = self.alphas_cumprod[t]

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def denoise_image(self, x_t, t, noise_unet):
        """
        Denoises an image x_t given the predicted noise (noise_unet) at time t.
        This is the reverse step.
        """
        if t.ndim == 1:
            alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        else:
            alpha_bar_t = self.alphas_cumprod[t]

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x_0 = (x_t - sqrt_one_minus_alpha_bar_t * noise_unet) / sqrt_alpha_bar_t

        return x_0

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding for time steps.
    Transforms a scalar time step 't' into a high-dimensional vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.embeddings = math.log(10000) / (self.half_dim - 1)
        self.embeddings = torch.exp(torch.arange(self.half_dim) * -self.embeddings)

    def forward(self, time):
        time_embedding = time.unsqueeze(1) * self.embeddings.to(time.device)

        time_embedding = torch.cat((time_embedding.sin(), time_embedding.cos()), dim=-1)
        return time_embedding

class TimeEmbeddingMLP(nn.Module):
    """
    MLP to process the sinusoidal time embedding into scale and shift parameters.
    """
    def __init__(self, dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization (AdaGN) layer.
    Applies GroupNorm, then scales and shifts features based on conditioning (time embedding).
    """
    def __init__(self, num_groups, num_channels, time_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.time_proj = nn.Linear(time_emb_dim, 2 * num_channels)

    def forward(self, x, time_emb):

        normed_x = self.norm(x)

        scale_shift = self.time_proj(time_emb)
        scale, shift = scale_shift.chunk(2, dim=1)

        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        output = normed_x * (1 + scale) + shift
        return output

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.norm1 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        self.use_dropout = use_dropout

    def forward(self, x, time_emb):
        identity = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x, time_emb)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, time_emb)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        x = self.downsample(x + identity)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_dropout=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.activation = nn.SiLU()
        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        self.use_dropout = use_dropout

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, skip_con_x, time_emb):
        x = self.upsample(x)

        if x.shape != skip_con_x.shape:
             x = torch.nn.functional.interpolate(x, size=skip_con_x.shape[2:], mode='nearest')

        x = torch.cat([x, skip_con_x], dim=1)

        identity = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x, time_emb)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, time_emb)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x + identity

class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=64, time_emb_dim=256): # Added time_emb_dim
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.time_emb_dim = time_emb_dim


        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)


        self.contract1 = ContractingBlock(hidden_channels, hidden_channels * 2, time_emb_dim)
        self.contract2 = ContractingBlock(hidden_channels * 2, hidden_channels * 4, time_emb_dim)
        self.contract3 = ContractingBlock(hidden_channels * 4, hidden_channels * 8, time_emb_dim)
        self.contract4 = ContractingBlock(hidden_channels * 8, hidden_channels * 16, time_emb_dim)
        self.contract5 = ContractingBlock(hidden_channels * 16, hidden_channels * 32, time_emb_dim)
        self.contract6 = ContractingBlock(hidden_channels * 32, hidden_channels * 64, time_emb_dim)


        self.expand0 = ExpandingBlock(hidden_channels * 64, hidden_channels * 32, time_emb_dim)
        self.expand1 = ExpandingBlock(hidden_channels * 32, hidden_channels * 16, time_emb_dim)
        self.expand2 = ExpandingBlock(hidden_channels * 16, hidden_channels * 8, time_emb_dim)
        self.expand3 = ExpandingBlock(hidden_channels * 8, hidden_channels * 4, time_emb_dim)
        self.expand4 = ExpandingBlock(hidden_channels * 4, hidden_channels * 2, time_emb_dim)
        self.expand5 = ExpandingBlock(hidden_channels * 2, hidden_channels, time_emb_dim)


        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x, time):
        time_emb = self.time_mlp(time)


        x0 = self.upfeature(x)
        x1 = self.contract1(x0, time_emb)
        x2 = self.contract2(x1, time_emb)
        x3 = self.contract3(x2, time_emb)
        x4 = self.contract4(x3, time_emb)
        x5 = self.contract5(x4, time_emb)
        x6 = self.contract6(x5, time_emb)

        x7 = self.expand0(x6, x5, time_emb)
        x8 = self.expand1(x7, x4, time_emb)
        x9 = self.expand2(x8, x3, time_emb)
        x10 = self.expand3(x9, x2, time_emb)
        x11 = self.expand4(x10, x1, time_emb)
        x12 = self.expand5(x11, x0, time_emb)

        xn = self.downfeature(x12)

        return xn

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, X):

        X = (X + 1) / 2
        X = self.normalize(X)

        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self,layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class DDIM():

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

    self.input_viton = ((warping(person_img, cloth_img) - 0.5)/0.5).to(device)
    self.agnostic_mask = self.transform(warping.agnostic_mask).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

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

        # if (i+1) % 10 == 0:

        #   plt.figure(figsize=(3, 3))
        #   plt.imshow((x_0_pred[0].permute(1, 2, 0).cpu().numpy() + 1)/2)
        #   plt.title(f"DDIM Step {i+1} (t={t.item()})")
        #   plt.axis('off')
        #   plt.show()

    return x_0_pred

