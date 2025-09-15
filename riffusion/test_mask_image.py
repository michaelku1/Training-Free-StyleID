import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np

def preprocess_mask(input_path, save_path, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """

    mask = Image.open(input_path)

    # Convert to grayscale
    mask = mask.convert("L")

    # Resize to integer multiple of 32
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.Resampling.NEAREST)

    # Convert to numpy array and rescale
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # # Tile and transpose
    # mask_np = np.tile(mask_np, (4, 1, 1))
    # mask_np = mask_np[None].transpose(0, 1, 2, 3)  # what does this step do?

    # Invert to repaint white and keep black
    # mask_np = 1 - mask_np

    # Convert numpy array to PIL Image
    img = torch.from_numpy(mask_np)

    save_image(img, save_path)

    return


INPUT_PATH = "/home/mku666/riffusion-hobby/results/riffusion_seed_mask_images/EGDB_DI_1/chopper/1.png"
SAVE_PATH = "/home/mku666/riffusion-hobby/masks/EGDB_DI_1_chopper_as_is.png"

preprocess_mask(INPUT_PATH, SAVE_PATH)


