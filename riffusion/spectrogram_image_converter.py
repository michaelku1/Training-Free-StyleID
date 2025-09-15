import numpy as np
import pydub
from PIL import Image
import os
import glob
import argparse

try:
    from riffusion.spectrogram_converter import SpectrogramConverter
    from riffusion.spectrogram_params import SpectrogramParams
    from riffusion.util import image_util

except ImportError:
    print("Using local files")
    from spectrogram_converter import SpectrogramConverter
    from spectrogram_params import SpectrogramParams
    from util import image_util


class SpectrogramImageConverter:
    """
    Convert between spectrogram images and audio segments.

    This is a wrapper around SpectrogramConverter that additionally converts from spectrograms
    to images and back. The real audio processing lives in SpectrogramConverter.
    """

    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        
        self.p = params
        self.device = device
        self.converter = SpectrogramConverter(params=params, device=device)

    def spectrogram_image_from_audio(
        self,
        segment: pydub.AudioSegment,
    ) -> Image.Image:
        """
        Compute a spectrogram image from an audio segment.

        Args:
            segment: Audio segment to convert

        Returns:
            Spectrogram image (in pillow format)
        """

        # assert int(segment.frame_rate) == self.p.sample_rate, "Sample rate mismatch"

        # TODO add resampling
        if int(segment.frame_rate) != self.p.sample_rate:
            print(f"Resampling from {segment.frame_rate}Hz to {self.p.sample_rate}Hz")
            segment = segment.set_frame_rate(self.p.sample_rate)

        # audio, sr = torchaudio.load(segment.filename)
        # print(f"Loaded audio from {sr}Hz")

        # audio = torchaudio.functional.resample(audio, sr, self.p.sample_rate)
        # torchaudio.save(segment.filename, audio, self.p.sample_rate)

        if self.p.stereo:
            if segment.channels == 1:
                print("WARNING: Mono audio but stereo=True, cloning channel")
                segment = segment.set_channels(2)
            elif segment.channels > 2:
                print("WARNING: Multi channel audio, reducing to stereo")
                segment = segment.set_channels(2)
        else:
            if segment.channels > 1:
                print("WARNING: Stereo audio but stereo=False, setting to mono")
                segment = segment.set_channels(1)

        spectrogram = self.converter.spectrogram_from_audio(segment)

        image = image_util.image_from_spectrogram(
            spectrogram,
            power=self.p.power_for_image,
        )

        # Store conversion params in exif metadata of the image
        exif_data = self.p.to_exif()
        exif_data[SpectrogramParams.ExifTags.MAX_VALUE.value] = float(np.max(spectrogram))
        exif = image.getexif()
        exif.update(exif_data.items())

        return image

    def audio_from_spectrogram_image(
        self,
        image: Image.Image,
        apply_filters: bool = True,
        max_value: float = 30e6,
    ) -> pydub.AudioSegment:
        """
        Reconstruct an audio segment from a spectrogram image.

        Args:
            image: Spectrogram image (in pillow format)
            apply_filters: Apply post-processing to improve the reconstructed audio
            max_value: Scaled max amplitude of the spectrogram. Shouldn't matter.
        """
        spectrogram = image_util.spectrogram_from_image(
            image,
            max_value=max_value,
            power=self.p.power_for_image,
            stereo=self.p.stereo,
        )

        segment = self.converter.audio_from_spectrogram(
            spectrogram,
            apply_filters=apply_filters,
        )

        return segment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spectrogram images from audio files")
    
    # Required arguments
    parser.add_argument("--source-type", choices=["AudioDI", "Tone"], required=True,
                       help="Audio source type: 'AudioDI' for clean DI recordings or 'Tone' for processed audio")
    parser.add_argument("--di", required=True,
                       help="DI selection (e.g., 'DI_1', 'DI_10', 'DI_100')")
    
    # Optional arguments
    parser.add_argument("--tone", 
                       help="Tone selection (only required when source-type is 'Tone'). "
                            "Available: Chopper, Easy Blues, First Compression, Gravity, "
                            "Light House, Moore Clean, New Guitar Icon, Rhapsody, Room 808, Dark Soul")
    parser.add_argument("--audio-base-path", 
                       default="/home/mku666/riffusion-hobby/sample_data/fx_data/EGDB-Large-Subset",
                       help="Base path to audio data directory")
    parser.add_argument("--output-base-path", 
                       default="/home/mku666/riffusion-hobby/results/spectrogram_images",
                       help="Base path for output spectrogram images")
    parser.add_argument("--device", default="cuda",
                       help="Device to use for processing (cuda/cpu)")
    parser.add_argument("--max-files", type=int,
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--file-list", 
                       help="Path to file containing list of specific files to process (one per line)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source_type == "Tone" and not args.tone:
        parser.error("--tone is required when --source-type is 'Tone'")
    
    # Build audio directory path based on source type
    if args.source_type == "AudioDI":
        audio_dir = f"{args.audio_base_path}/AudioDI/{args.di}"
        output_subdir = f"AudioDI_{args.di}"
    elif args.source_type == "Tone":
        audio_dir = f"{args.audio_base_path}/Tone/{args.tone}/{args.di}"
        output_subdir = f"Tone_{args.tone}_{args.di}"
    else:
        raise ValueError("source_type must be either 'AudioDI' or 'Tone'")
    
    # Create output directory
    output_path = f"{args.output_base_path}/{output_subdir}"
    os.makedirs(output_path, exist_ok=True)
    
    # Get files to process
    if args.file_list:
        # Read specific files from list
        with open(args.file_list, 'r') as f:
            file_names = [line.strip() for line in f if line.strip()]
        wav_files = [f"{audio_dir}/{name}" for name in file_names if os.path.exists(f"{audio_dir}/{name}")]
    else:
        # Get all .wav files in the directory
        wav_files = glob.glob(f"{audio_dir}/*.wav")
        wav_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # Limit files if max_files is specified
    if args.max_files:
        wav_files = wav_files[:args.max_files]
    
    print(f"Processing {len(wav_files)} audio files from: {audio_dir}")
    print(f"Output directory: {output_path}")
    print(f"Device: {args.device}")
    
    # Initialize spectrogram image converter once
    params = SpectrogramParams()
    image_converter = SpectrogramImageConverter(params=params, device=args.device)
    
    # Process each audio file
    processed_count = 0
    error_count = 0
    
    for wav_file in wav_files:
        try:
            # Get the filename without extension for output naming
            filename = os.path.splitext(os.path.basename(wav_file))[0]
            
            # Load audio
            audio_segment = pydub.AudioSegment.from_file(wav_file)
            
            # Convert audio to spectrogram image
            image = image_converter.spectrogram_image_from_audio(audio_segment)
            
            # Save image
            output_file = f"{output_path}/{filename}.png"
            image.save(output_file)
            
            processed_count += 1
            print(f"Processed ({processed_count}/{len(wav_files)}): {filename}.wav -> {filename}.png")
            
        except Exception as e:
            error_count += 1
            print(f"Error processing {wav_file}: {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors: {error_count} files")
    print(f"Generated spectrogram images saved to: {output_path}")
        
        
        
        
        
        