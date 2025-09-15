#!/bin/bash

# Simple spectrogram generator
# Usage: ./generate_spectrograms.sh [source_type] [di] [tone] [max_files]

SOURCE_TYPE=${1:-"Tone"}
DI=${2:-"DI_1"}
TONE=${3:-"Chopper"}
MAX_FILES=${4:-5}

AUDIO_BASE="/home/mku666/riffusion-hobby/sample_data/fx_data/EGDB-Large-Subset"
OUTPUT_BASE="/home/mku666/riffusion-hobby/results/spectrogram_images"

# Build paths
if [ "$SOURCE_TYPE" = "AudioDI" ]; then
    AUDIO_DIR="$AUDIO_BASE/AudioDI/$DI"
    OUTPUT_DIR="$OUTPUT_BASE/AudioDI_$DI"
else
    AUDIO_DIR="$AUDIO_BASE/Tone/$TONE/$DI"
    OUTPUT_DIR="$OUTPUT_BASE/Tone_${TONE}_$DI"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Python script
python riffusion/spectrogram_image_converter.py \
    --source-type "$SOURCE_TYPE" \
    --di "$DI" \
    --tone "$TONE" \
    --max-files "$MAX_FILES" \
    --audio-base-path "$AUDIO_BASE" \
    --output-base-path "$OUTPUT_BASE" 