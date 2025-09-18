"""
Flask server that serves the riffusion model as an API.
"""

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

from riffusion.datatypes import InferenceInput, InferenceOutput

# NOTE riffusion pipeline with only one input (no interpolation)
# from riffusion.riffusion_pipeline_mix_tone import RiffusionPipeline
from riffusion.riffusion_pipeline import RiffusionPipeline


from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import base64_util


# Flask app with CORS
app = flask.Flask(__name__)
CORS(app)

# Log at the INFO level to both stdout and disk
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("server.log"))

# Global variable for the model pipeline
# PIPELINE: T.Optional[RiffusionPipeline] = None

# Where built-in seed images are storedrun
def run_app(
    *,
    checkpoint: str = "riffusion/riffusion-model-v1",
    no_traced_unet: bool = False,
    device: str = "cuda",
    host: str = "127.0.0.1",
    port: int = 3013,
    debug: bool = False,
    ssl_certificate: T.Optional[str] = None,
    ssl_key: T.Optional[str] = None,
):
    """
    Run a flask API that serves the given riffusion model checkpoint.
    """
    # Initialize the model
    global PIPELINE

    PIPELINE = RiffusionPipeline.load_checkpoint(
        checkpoint=checkpoint,
        use_traced_unet=True,
        device=device,
    )
    
    args = dict(
        debug=debug,
        threaded=False,
        host=host,
        port=port,
    )

    if ssl_certificate:
        assert ssl_key is not None
        args["ssl_context"] = (ssl_certificate, ssl_key)

    app.run(**args)  # type: ignore


@app.route("/run_inference/", methods=["POST"])
def run_inference():
    """
    Execute the riffusion model as an API.

    Inputs:
        Serialized JSON of the InferenceInput dataclass

    Returns:
        Serialized JSON of the InferenceOutput dataclass
    """
    start_time = time.time()

    # Parse the payload as JSON
    json_data = json.loads(flask.request.data)

    # Log the request
    logging.info(json_data)

    # Parse an InferenceInput dataclass from the payload
    try:
        inputs = dacite.from_dict(InferenceInput, json_data)
    except dacite.exceptions.WrongTypeError as exception:
        logging.info(json_data)
        return str(exception), 400
    except dacite.exceptions.MissingValueError as exception:
        logging.info(json_data)
        return str(exception), 400

    # NOTE
    response = compute_request(
        inputs=inputs,
        pipeline=PIPELINE,
    )

    # Log the total time
    logging.info(f"Request took {time.time() - start_time:.2f} s")

    return response


def compute_request(
    inputs: InferenceInput,
    pipeline: None,
) -> T.Union[str, T.Tuple[str, int]]:
    """
    Does all the heavy lifting of the request.

    Args:
        inputs: The input dataclass
        pipeline: The riffusion model pipeline
    """
    
    # Load the seed image by ID
    init_image_path = Path(f"{inputs.seed_image_path}.png")

    print("######################### input image path: ", init_image_path)

    if not init_image_path.is_file():
        return f"Invalid seed image: {inputs.seed_image_path}", 400
    init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    mask_image: T.Optional[PIL.Image.Image] = None

    # NOTE pass mask image here
    # mask_image = PIL.Image.open("...png").convert("RGB")
    # NOTE pass mask image here
    # mask_image = PIL.Image.open("...png").convert("RGB")
    if inputs.mask_image_path:
        print(f"DEBUG: mask_image_path type: {type(inputs.mask_image_path)}")
        print(f"DEBUG: mask_image_path value: {inputs.mask_image_path}")
        
        # multi style mask
        if isinstance(inputs.mask_image_path, list):
            print("DEBUG: Processing as list of masks")
            mask_image = []
            for i, mask_path in enumerate(inputs.mask_image_path):
                mask_image_path = Path(f"{mask_path}.png")
                print(f"DEBUG: Checking mask {i}: {mask_image_path}")
                if not mask_image_path.is_file():
                    return f"Invalid mask image: {mask_path}", 400
                mask_image.append(PIL.Image.open(str(mask_image_path)).convert("RGB"))
        else:
            # single style mask
            print("DEBUG: Processing as single mask")
            mask_image_path = Path(f"{inputs.mask_image_path}.png")
            print(f"DEBUG: Checking single mask: {mask_image_path}")
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
    torch.cuda.ipc_collect()  # (optional) reclaim inter-process memorys


    clean_to_style_output_name = f"{''.join(inputs.seed_image_path.split('/')[-2:])}_to_{''.join(inputs.mask_image_path[0].split('/')[-2:])}"
    for i, mask_image_path in enumerate(inputs.mask_image_path):
        if i == 0:
            final_output_name = clean_to_style_output_name
        else:
            final_output_name = f"{final_output_name}_and_{''.join(mask_image_path.split('/')[-2:])}" + "_" + str(i)
        
    with open(f"{inputs.output_path}/{final_output_name}.json", "w") as f:
        json.dump(dataclasses.asdict(output), f, indent=2, ensure_ascii=False)

    return output



if __name__ == "__main__":
    import argh

    argh.dispatch_command(run_app)
