import torch
import numpy as np
import PIL
from PIL import Image
from pathlib import Path
import torchvision

def preprocess_mask(mask: Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """
    # Convert to grayscale
    mask = mask.convert("L")

    # Resize to integer multiple of 32
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.Resampling.NEAREST)

    # Convert to numpy array and rescale
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Tile and transpose
    mask_np = np.tile(mask_np, (4, 1, 1))
    mask_np = mask_np[None].transpose(0, 1, 2, 3)  # what does this step do?

    # Invert to repaint white and keep black
    # BUG meant to keep white instead
    mask_np = 1 - mask_np

    return torch.from_numpy(mask_np)


if __name__ == "__main__":

    mask_image_path = Path("/home/mku666/riffusion-hobby/results/riffusion_seed_mask_images/EGDB_DI_1/chopper/1.png")
    mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")
    vae_config_block_out_channels_length = 4
    vae_scale_factor = 2 ** (vae_config_block_out_channels_length - 1)
    # rescale mask dimensions corresponding to vae dims
    mask = preprocess_mask(mask_image, scale_factor=vae_scale_factor)
    torchvision.utils.save_image(mask, "preprocess_mask.png")
