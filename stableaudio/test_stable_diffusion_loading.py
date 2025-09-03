import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler, DDIMScheduler


# From "https://huggingface.co/blog/stable_diffusion"
def load_stable_diffusion(sd_version='2.1', precision_t=torch.float32, device="cuda"):
    if sd_version == '2.1':
        model_key = "stabilityai/stable-diffusion-2-1"
    elif sd_version == '2.1-base':
        model_key = "stabilityai/stable-diffusion-2-1-base"
    elif sd_version == '2.0':
        model_key = "stabilityai/stable-diffusion-2-base"
    elif sd_version == '1.5':
        model_key = "runwayml/stable-diffusion-v1-5"
        
    # Create model
    pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)
    
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    # import xformer
    # unet.enable_xformers_memory_efficient_attention()
    
    del pipe
    
    # Use DDIM scheduler
    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)
    
    return vae, tokenizer, text_encoder, unet, scheduler

def decode_latent(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    return image

def encode_latent(images, vae):
    # encode the image with vae
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
    latents = 0.18215 * latents
    return latents


def test_stable_diffusion_loading():
    """
    Test stable diffusion loading
    """

    style_image = torch.randn(1, 3, 512, 512)
    dtype = torch.float32
    normalize = lambda x: x * 2.0 - 1.0

    # load stable diffusion
    vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(sd_version='2.1', precision_t=torch.float32, device="cuda")

    style_latent = encode_latent(normalize(style_image).to(device=vae.device, dtype=dtype), vae)


if __name__ == "__main__":
    test_stable_diffusion_loading()




