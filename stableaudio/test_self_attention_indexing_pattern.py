import torch
from diffusers import StableAudioPipeline

# Load the pipeline
pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0")

# Get the core transformer (DiT)
transformer = pipe.transformer  # This is DiT, like a ViT


# Typically inside the DiT, there's a .transformer or .blocks attribute
blocks = transformer.transformer_blocks  # This is a ModuleList of transformer blocks


# NOTE pipe.transformer.transformer_blocks
# Now let's index into the attention layers
for i, block in enumerate(blocks):
    self_attn = block.attn1
    print(f"Block {i} Self-Attention: {self_attn}")