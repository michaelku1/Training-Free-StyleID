##

import torch
import safetensors.torch
import json
from stable_audio_tools.models.factory import create_pretransform_from_config
from stable_audio_tools import get_pretrained_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Save pretransform
pretransform = model.pretransform
pretransform_state_dict = model.pretransform.state_dict()

file_path = 'pretransform.safetensors'
safetensors.torch.save_file(pretransform_state_dict, file_path)

# Load the pretransform configuration
import os
pretransform_config_path = os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--stable-audio-open-1.0/snapshots/f21265c1e2710b3bd2386596943f0007f55f802e/model_config.json")
with open(pretransform_config_path) as f:
    pretransform_config = json.load(f)


# Create the pretransform model from the configuration
reload_pretransform = create_pretransform_from_config(pretransform_config, sample_rate=model_config["sample_rate"])
reload_pretransform = reload_pretransform.to(device)

# Check if the original pretransform and the reloaded pretransform are of the same type
print(type(pretransform) == type(reload_pretransform))  # Should print True

# Apply the state dictionary to the pretransform model
state_dict = safetensors.torch.load_file(file_path)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('model.', '')  
    new_state_dict[new_key] = value

reload_pretransform.load_state_dict(new_state_dict)

print("output type", type(reload_pretransform))
print("model type", type(model))