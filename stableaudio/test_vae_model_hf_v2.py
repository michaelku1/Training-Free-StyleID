from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
import json
import torch
import os

from huggingface_hub import hf_hub_download

def get_vae_from_stable_audio_open_1_0():
    """
    Get the VAE from the Stable Audio Open 1.0 model.

    returns:
        vae: the VAE model
    """

    # download
    model_id = "stabilityai/stable-audio-open-1.0"
    model_config_path = hf_hub_download(repo_id=model_id, filename="model_config.json")
    model_ckpt_path = hf_hub_download(repo_id=model_id, filename="model.ckpt")

    # init stable-audio-open-1.0 and load the weights
    model = create_model_from_config(json.load(open(model_config_path)))
    copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    # save just vae to file
    vae_ckpt_path = "vae.ckpt"
    torch.save({'state_dict': model.pretransform.model.state_dict()}, vae_ckpt_path)

    # Init just pre-trained VAE. Need config file that you can find at link below:
    # https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json
    stable_audio_vae_config_path = os.path.expanduser("/home/mku666/riffusion-hobby/stableaudio/stable_audio_2_0_vae.json")
    vae = create_model_from_config(json.load(open(stable_audio_vae_config_path)))

    # load model weights
    copy_state_dict(vae, load_ckpt_state_dict(vae_ckpt_path))
    vae.to('cuda').eval().requires_grad_(False)

    return vae

