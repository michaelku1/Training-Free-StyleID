

## check cmds for various generation commands


## riffusion
generate spectrograms --> riffusion/spectrogram_image_converter.py

batch generate spectrograms --> generate_spectrograms.sh

```bash


First 10 files from Easy Blues tone, DI_1
./generate_spectrograms.sh Tone DI_1 "Easy Blues" 10

# All AudioDI files from DI_1
./generate_spectrograms.sh AudioDI DI_1 "" 999

# First 3 files from Moore Clean tone, DI_10
./generate_spectrograms.sh Tone DI_10 "Moore Clean" 3

# First 8 files from Gravity tone, DI_1
./generate_spectrograms.sh Tone DI_1 Gravity 8

```

riffusion server --> riffusion/server.py

generation --> curl -X POST http://127.0.0.1:8080/run_inference/ -H "Content-Type: application/json" -d '{"start":{"prompt":"","seed":42,"denoising":0.75,"guidance":7.0},"end":{"prompt":"","seed":123,"denoising":0.75,"guidance":7.0},"alpha":0.5,"num_inference_steps":200,"seed_image_id":"Chopper_egdb_1_spectrogram_image"}'

## batch processing
Run batch inference with custom DI directory and file count:

```bash
# Usage: ./batch_run.sh <DI_DIRECTORY> <NUM_FILES>
./batch_run.sh DI_1 10        # Process first 10 wav files from DI_1
./batch_run.sh DI_105 5       # Process first 5 wav files from DI_105
```

The script will:
- Use wav files from `/sample_data/fx_data/EGDB-Large-Subset/AudioDI/<DI_DIRECTORY>/` as seed audio
- Apply all available tone masks from `/sample_data/fx_data/EGDB-Large-Subset/Tone/` directory
- Generate combinations of each seed file with each tone variation
- Output results to `/results/audio/`

Available tone names: Moore Clean, Rhapsody, First Compression, New Guitar Icon, Gravity, Chopper, Room 808, Light House, Dark Soul, Easy Blues

## stable audio
TBD

## style injection
TBD
