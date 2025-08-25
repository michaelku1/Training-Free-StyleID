from riffusion.riffusion_pipeline import preprocess_mask
from PIL import Image
from torchvision import transforms

to_pil = transforms.ToPILImage()

mask = Image.open("/home/mku666/riffusion-hobby/riffusion/seed_images/mask_beat_lines_80.png")
mask = preprocess_mask(mask)

breakpoint()

pil_mask = to_pil(mask)
pil_mask.save("/home/mku666/riffusion-hobby/riffusion/mask_processed/mask_beat_lines_80_processed.png")

