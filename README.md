

## check cmds for various generation commands


## riffusion
generate spectrograms --> riffusion/spectrogram_image_converter.py

batch generate spectrograms --> generate_spectrograms.sh

```bash
# Generate spectrograms for all tones using DI_1
# Usage: ./generate_spectrograms.sh [max_files_per_tone] [audio_base_path]

# Default: 5 files from each tone
./generate_spectrograms.sh

# 2 files from each tone in EGDB-Large-Subset
./generate_spectrograms.sh 2 "/home/mku666/riffusion-hobby/sample_data/fx_data/EGDB-Large-Subset"

# 10 files from each tone in EGDB-Large/train
./generate_spectrograms.sh 10 "/home/mku666/riffusion-hobby/sample_data/EGDB-Large/train"
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



## spectrogram batch processing
```bash
# Usage: ./batch_run_spectrogram.sh <SEED_DIR> <MASK_DIR> <NUM_SEED_IMAGES>
./batch_run_spectrogram.sh AudioDI_DI_1 "Tone_Moore Clean_DI_1" 5
```

