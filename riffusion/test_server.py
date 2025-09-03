import dataclasses
import io
import json
import logging
import time
import typing as T
from pathlib import Path

import dacite
import flask
import PIL
import torch
from flask_cors import CORS

# Fix CUDA linear algebra backend to avoid cusolver errors
torch.backends.cuda.preferred_linalg_library('magma')

# NOTE original riffusion pipeline
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.datatypes import InferenceInput, InferenceOutput

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import base64_util

OUTPUT_DIR = "/home/mku666/riffusion-hobby/results/audio"


checkpoint = "riffusion/riffusion-model-v1"
device = "cuda"

PIPELINE = RiffusionPipeline.load_checkpoint(
    checkpoint=checkpoint,
    use_traced_unet=True,
    device=device,
)

def compute_request(
    inputs: InferenceInput,
    pipeline: RiffusionPipeline,
) -> T.Union[str, T.Tuple[str, int]]:
    """
    Does all the heavy lifting of the request.

    Args:
        inputs: The input dataclass
        pipeline: The riffusion model pipeline
    """
    # Load the seed image by ID
    init_image_path = Path(f"{inputs.seed_image_path}")

    print("######################### input image path: ", init_image_path)

    if not init_image_path.is_file():
        return f"Invalid seed image: {inputs.seed_image_path}", 400
    init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    mask_image: T.Optional[PIL.Image.Image] = None

    # NOTE pass mask image here
    # mask_image = PIL.Image.open("...png").convert("RGB")
    if inputs.mask_image_path:
        mask_image_path = Path(f"{inputs.mask_image_path}")
        if not mask_image_path.is_file():
            return f"Invalid mask image: {inputs.mask_image_path}", 400
        mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")

    # Execute the model to get the spectrogram image
    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=mask_image,
    )

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    # Reconstruct audio from the image
    # TODO(hayk): It may help performance a bit to cache this object
    # Use CPU for audio processing to avoid CUDA solver issues
    converter = SpectrogramImageConverter(params=params, device="cpu")

    # NOTE 轉回 audio signal
    segment = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
    )

    # Export audio to MP3 bytes
    mp3_bytes = io.BytesIO()
    segment.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)

    # Export image to JPEG bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, exif=image.getexif(), format="JPEG")
    image_bytes.seek(0)

    # Assemble the output dataclass
    output = InferenceOutput(
        image="data:image/jpeg;base64," + base64_util.encode(image_bytes),
        audio="data:audio/mpeg;base64," + base64_util.encode(mp3_bytes),
        duration_s=segment.duration_seconds,
    )

    # release memory
    import gc
    del image, mask_image, init_image  # delete big tensors
    gc.collect()
    torch.cuda.empty_cache()  # free cached memory
    torch.cuda.ipc_collect()  # (optional) reclaim inter-process memory
    
    output_path = f"{inputs.seed_image_path.split('/')[-2]}_{inputs.mask_image_path.split('/')[-2]}"

    with open(f"{OUTPUT_DIR}/{output_path}.json", "w") as f:
        json.dump(dataclasses.asdict(output), f, indent=2, ensure_ascii=False)

    return output


if __name__ == "__main__":
    import json
    import sys

    json_str = sys.argv[1]

    json_data = json.loads(json_str)

    inputs = dacite.from_dict(InferenceInput, json_data)

    response = compute_request(
        inputs=inputs,
        pipeline=PIPELINE,
    )
