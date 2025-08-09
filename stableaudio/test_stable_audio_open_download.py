import torch
import torchaudio
from transformers import T5EncoderModel, T5Tokenizer
# from stableaudio.stable_audio_pipeline import DiffusionTransformer
from diffusers import DDPMScheduler


import encodec
import k_diffusion as K

HF_ACCESS_TOKEN = open("hf_access_token", "r").read()

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Login to Hugging Face (uncomment and run once to authenticate)
    from huggingface_hub import login
    login("HF_ACCESS_TOKEN")  # This will prompt for your token
    print("successfully logged in!")
except Exception as e:
    print(f"Error logging in: {e}")
    print("Make sure you have logged in to Hugging Face with: huggingface-cli login")


# check stable-audio-open-v1.0 model
# try:
#     model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
#     print("Successfully loaded stable-audio-open-1.0 model")

# test model loading
try:
    checkpoint = "stabilityai/stable-audio-open-1.0"
    text_encoder = T5EncoderModel.from_pretrained(checkpoint, subfolder="text_encoder") # ok
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, subfolder="tokenizer") # ok
    scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder="scheduler")


except Exception as e:
    print(e)
