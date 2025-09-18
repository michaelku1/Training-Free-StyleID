

## check cmds for various generation commands


## riffusion
generate spectrograms --> riffusion/spectrogram_image_converter.py

batch generate spectrograms --> bash_scripts/generate_spectrograms.sh

```bash
# Generate spectrograms for instrument directories - works with three-level structure
# Usage: ./bash_scripts/generate_spectrograms.sh [max_files_per_instrument] [instrument_path]

# Process 5 files from accordion instrument
./bash_scripts/generate_spectrograms.sh 5 ./riffusion-hobby/data_acoustic/musicTI/accordion

# Process 3 files from violin instrument  
./bash_scripts/generate_spectrograms.sh 3 ./riffusion-hobby/data_acoustic/musicTI/violin

# Use default settings (5 files from accordion)
./bash_scripts/generate_spectrograms.sh

# Results are saved to: ./riffusion-hobby/results/spectrogram_images/Tone_{instrument}_files/
```

riffusion server --> riffusion/server.py

generation --> curl -X POST http://127.0.0.1:8080/run_inference/ -H "Content-Type: application/json" -d '{"start":{"prompt":"","seed":42,"denoising":0.75,"guidance":7.0},"end":{"prompt":"","seed":123,"denoising":0.75,"guidance":7.0},"alpha":0.5,"num_inference_steps":200,"seed_image_id":"Chopper_egdb_1_spectrogram_image"}'

## batch processing
The spectrogram generation script supports three-level directory structures like:
- `/data_acoustic/musicTI/accordion/`
- `/data_acoustic/musicTI/violin/`

The script will:
- Process all .wav files directly from the instrument directory
- Generate spectrograms using CPU processing (to avoid CUDA memory issues)
- Create temporary directory structure for compatibility with existing Python scripts
- Output spectrogram images to `/results/spectrogram_images/Tone_{instrument}_files/`

## stable audio
TBD

## style injection
TBD

```

