import torch
import torchaudio
import pydub
import numpy as np
from diffusers import AutoencoderKL
from PIL import Image
from diffusers.pipelines.pipeline_utils import numpy_to_pil

try:
    from riffusion.spectrogram_params import SpectrogramParams
    from riffusion.spectrogram_converter import SpectrogramConverter
    from riffusion.spectrogram_image_converter import SpectrogramImageConverter
except ImportError:
    print("Using local files")
    from spectrogram_params import SpectrogramParams
    from spectrogram_converter import SpectrogramConverter
    from spectrogram_image_converter import SpectrogramImageConverter

def spectrogram_to_audio(spectrogram_image: Image.Image, output_path: str):
    """
    Convert spectrogram image back to audio.
    
    Args:
        spectrogram_image: PIL Image of the spectrogram
        output_path: Path to save the output audio
    """
    print(f"Converting spectrogram to audio: {output_path}")

    # NOTE params and image_converter should be passed as arguments to both the spectrogram_to_audio function and the audio_to_spectrogram function
    params = SpectrogramParams()
    image_converter = SpectrogramImageConverter(params=params, device="cuda")
    
    # Convert image back to audio
    audio_segment = image_converter.audio_from_spectrogram_image(spectrogram_image)
        
    # Save audio to file
    audio_segment.export(output_path, format="wav")

def audio_to_spectrogram(audio_path: str) -> Image.Image:
    """
    Convert audio file to spectrogram image.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        PIL Image of the spectrogram
    """
    print(f"Converting audio to spectrogram: {audio_path}")

    # NOTE params and image_converter should be passed as arguments to both the spectrogram_to_audio function and the audio_to_spectrogram function
    params = SpectrogramParams()
    image_converter = SpectrogramImageConverter(params=params, device="cuda")
    
    # Load audio using pydub
    audio_segment = pydub.AudioSegment.from_file(audio_path)

    # Convert audio to spectrogram image
    image = image_converter.spectrogram_image_from_audio(audio_segment)
    
    return image

def test_riffusion_vae():
    # Load audio file
    audio_path = "/home/mku666/riffusion-hobby/stable_audio_api/sample_data/fx_data/EGDB-Large-Subset/Tone/Chopper/DI_1/1.wav"
    
    # Load audio with torchaudio
    # waveform, sample_rate = torchaudio.load(audio_path)
    # print(f"Original audio: {waveform.shape}, {sample_rate}Hz")
    image = audio_to_spectrogram(audio_path)

    mel_spec_processed = preprocess_image(image)
    mel_spec_processed = mel_spec_processed.to("cuda")

    print(f"Mel spectrogram shape: {mel_spec_processed.shape}")
    
    # Load Riffusion VAE from Hugging Face
    print("Loading Riffusion VAE...")
    vae = AutoencoderKL.from_pretrained("riffusion/riffusion-model-v1", subfolder="vae")
    vae = vae.to("cuda")
    vae.eval()
    
    # Encode to latent space
    print("Encoding to latent space...")
    with torch.no_grad():
        latents = vae.encode(mel_spec_processed).latent_dist.sample()
        print(f"Latent shape: {latents.shape}")
    
    # Decode from latent space
    print("Decoding from latent space...")
    with torch.no_grad():
        latents = 1.0 / 0.18215 * latents
        reconstructed_spec = vae.decode(latents).sample
        print(f"Reconstructed spectrogram shape: {reconstructed_spec.shape}")

    # Convert back to audio
    print("Converting spectrogram back to audio...")

    reconstructed_spec = (reconstructed_spec / 2 + 0.5).clamp(0, 1)
    reconstructed_spec = reconstructed_spec.cpu().permute(0, 2, 3, 1).numpy()

    reconstructed_spec_np = reconstructed_spec.squeeze()

    # convert to pil image
    # BUG Image.fromarray is very very buggy, its not very pytorch friendly
    # reconstructed_spec_pil = Image.fromarray(reconstructed_spec_np, mode="L")
    reconstructed_spec_pil = numpy_to_pil(reconstructed_spec_np)[0]
    spectrogram_to_audio(reconstructed_spec_pil, output_path="reconstructed_audio_vae.wav")
    
    print("Files saved: reconstructed_audio_vae.wav")
    print("Listen to the file to check for VAE artifacts!")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    # NOTE supposedly dimension should be (512, 1696, 3)
    image_np = np.array(image).astype(np.float32) / 255.0

    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)

    return 2.0 * image_torch - 1.0

if __name__ == "__main__":
    test_riffusion_vae() 