import pydub

try:
    from riffusion.spectrogram_image_converter import SpectrogramImageConverter
    from riffusion.spectrogram_params import SpectrogramParams
except ImportError:
    print("Using local files")
    from spectrogram_image_converter import SpectrogramImageConverter
    from spectrogram_params import SpectrogramParams


def test_mel_roundtrip():
    # Load audio file
    audio_path = "/home/mku666/riffusion-hobby/stable_audio_api/sample_data/fx_data/EGDB-Large-Subset/Tone/Chopper/DI_1/1.wav"
    original_audio = pydub.AudioSegment.from_file(audio_path)
    
    print(f"Original audio: {len(original_audio)}ms, {original_audio.frame_rate}Hz, {original_audio.channels} channels")
    
    # Initialize converter
    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params=params, device="cuda")
    
    # Convert audio to mel spectrogram
    print("Converting audio to mel spectrogram...")
    mel_spec = converter.converter.spectrogram_from_audio(original_audio)
    print(f"Mel spectrogram shape: {mel_spec.shape}")

    
    # Convert mel spectrogram back to audio
    print("Converting mel spectrogram back to audio...")
    reconstructed_audio = converter.converter.audio_from_spectrogram(mel_spec, apply_filters=True)
    print(f"Reconstructed audio: {len(reconstructed_audio)}ms, {reconstructed_audio.frame_rate}Hz, {reconstructed_audio.channels} channels")
    
    # Save both for comparison
    original_audio.export("original_audio.wav", format="wav")
    reconstructed_audio.export("reconstructed_audio.wav", format="wav")
    
    print("Files saved: original_audio.wav, reconstructed_audio.wav")
    print("Listen to both files to check for artifacts!")

if __name__ == "__main__":
    test_mel_roundtrip() 