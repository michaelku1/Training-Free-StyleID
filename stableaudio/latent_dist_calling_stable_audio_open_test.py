import torch
from diffusers.models import AutoencoderKL
import torchaudio

vae = AutoencoderKL.from_pretrained("stable-audio-open-1.0", subfolder="vae")

waveform, sr = torchaudio.load("example.wav")
spectrogram = torchaudio.transforms.MelSpectrogram(sr)(waveform)
spectrogram = spectrogram.unsqueeze(0)
spectrogram = spectrogram / spectrogram.max()

outputs = vae.encode(spectrogram)
latent_dist = outputs.latent_dist
latent = latent_dist.sample()

reconstructed_spectrogram = vae.decode(latent).sample
print(latent.shape)