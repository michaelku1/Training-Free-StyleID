import numpy as np
import pydub
from PIL import Image

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
    # style_names = ["Chopper", "Easy Blues", "First Compression", "Gravity", "Light House", "Moore Clean", "New Guitar Icon", "Rhapsody", "Room 808"]
    # style_names = ["Chopper"]

    # for style_name in style_names:
        # load audio
        # audio_path = f"/home/mku666/riffusion-hobby/stable_audio_api/sample_data/fx_data/EGDB-Large-Subset/Tone/{style_name}/DI_1/1.wav"
        # audio_segment = pydub.AudioSegment.from_file(audio_path)

        # # initialize spectrogram image converter
        # params = SpectrogramParams()
        # image_converter = SpectrogramImageConverter(params=params, device="cuda")

        # # convert audio to spectrogram image
        # image = image_converter.spectrogram_image_from_audio(audio_segment)

        # # NOTE convert to numpy array
        # image_array = np.array(image)
        # print(image_array.shape)
        
        # save image 
        # image.save(f"./riffusion/test_spec_images/{style_name}_egdb_1_spectrogram_image.png")
    
    # audio_clip_names = ["1", "2", "3"]
    audio_clip_names = ["1", "2", "3"]

    for audio_clip_name in audio_clip_names:

        # audio_path = f"/home/mku666/riffusion-hobby/stable_audio_api/sample_data/fx_data/EGDB-Large-Subset/AudioDI/DI_1/{audio_clip_name}.wav"
        # audio_path = f"/home/mku666/riffusion-hobby/stable_audio_api/sample_data/accordion/accordion{audio_clip_name}.wav"
        audio_path = f"/home/mku666/riffusion-hobby/stable_audio_api/sample_data/violin/violin{audio_clip_name}.wav"
        # audio_path = f"/home/mku666/riffusion-hobby/stable_audio_api/sample_data/piano/piano{audio_clip_name}.wav"

        # NOTE egdb DI images
        audio_segment = pydub.AudioSegment.from_file(audio_path)

        # initialize spectrogram image converter
        params = SpectrogramParams()
        image_converter = SpectrogramImageConverter(params=params, device="cuda")

        # convert audio to spectrogram image
        image = image_converter.spectrogram_image_from_audio(audio_segment)

        # NOTE convert to numpy array
        image_array = np.array(image)

        # save image
        image.save(f"./violin123/{audio_clip_name}.png")
        
        
        
        
        
        