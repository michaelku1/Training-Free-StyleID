from diffusers.models import AutoencoderKL
import torch

"""
procedure:
1. encode audio tensor to latent space, returns a DiagonalGaussianDistribution object
2. sample by using mean of and std of image and gaussian noise
"""

audio_tensor = torch.randn(1, 3, 64, 64)

vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
outputs = vae.encode(audio_tensor, return_dict=True)

# NOTE latent_dist represents a Gaussian distribution for each latent pixel
latent_dist = outputs.latent_dist # <diffusers.models.autoencoders.vae.DiagonalGaussianDistribution object at 0x76353cfc6de0>
# NOTE 
latent_dist_sample = latent_dist.sample() # torch.Size([1, 4, 8, 8])





