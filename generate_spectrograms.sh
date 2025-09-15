#!/bin/bash

# Generate spectrograms for all tones using DI_1 - works with both directory structures
# Usage: ./generate_spectrograms.sh [max_files_per_tone] [audio_base_path]

MAX_FILES=${1:-5}
AUDIO_BASE=${2:-"/home/mku666/riffusion-hobby/sample_data/EGDB-Large/train"}
DI="DI_1"

echo "=== GENERATING SPECTROGRAMS FOR ALL TONES ==="
echo "Audio Base Path: $AUDIO_BASE"
echo "DI: $DI"
echo "Max Files per Tone: $MAX_FILES"
echo ""

# Auto-detect directory structure
if [ -d "$AUDIO_BASE/Tone" ]; then
    STRUCTURE="subset"
    TONE_BASE="$AUDIO_BASE/Tone"
elif [ -d "$AUDIO_BASE/Chopper" ] || [ -d "$AUDIO_BASE/AudioDI" ]; then
    STRUCTURE="large"
    TONE_BASE="$AUDIO_BASE"
else
    echo "ERROR: Cannot detect directory structure in $AUDIO_BASE"
    exit 1
fi

echo "Detected structure: $STRUCTURE"
echo "Tone base directory: $TONE_BASE"
echo ""

# Get all tone directories
TONE_DIRS=()
while IFS= read -r -d '' dir; do
    TONE_DIRS+=("$dir")
done < <(find "$TONE_BASE" -maxdepth 1 -type d -not -name "AudioDI" -not -path "$TONE_BASE" -print0)

TOTAL_TONES=${#TONE_DIRS[@]}
CURRENT_TONE=0

echo "Found $TOTAL_TONES tones to process:"
for tone in "${TONE_DIRS[@]:0:10}"; do
    echo "  - $(basename "$tone")"
done
if [ $TOTAL_TONES -gt 10 ]; then
    echo "  ... and $((TOTAL_TONES - 10)) more"
fi
echo ""

# Process each tone
for TONE_DIR in "${TONE_DIRS[@]}"; do
    TONE=$(basename "$TONE_DIR")
    CURRENT_TONE=$((CURRENT_TONE + 1))
    
    echo "Processing tone $CURRENT_TONE/$TOTAL_TONES: $TONE"
    
    # Check if DI directory exists for this tone
    if [ ! -d "$TONE_DIR/$DI" ]; then
        echo "  Skipping $TONE - no $DI directory found"
        continue
    fi
    
    # Count available files
    AVAILABLE_FILES=$(ls "$TONE_DIR/$DI"/*.wav 2>/dev/null | wc -l)
    if [ $AVAILABLE_FILES -eq 0 ]; then
        echo "  Skipping $TONE - no .wav files found in $DI"
        continue
    fi
    
    FILES_TO_PROCESS=$((AVAILABLE_FILES < MAX_FILES ? AVAILABLE_FILES : MAX_FILES))
    echo "  Available files: $AVAILABLE_FILES, Processing: $FILES_TO_PROCESS"
    
    # Run the spectrogram generation
    python riffusion/spectrogram_image_converter_flexible.py \
        --source-type "Tone" \
        --di "$DI" \
        --tone "$TONE" \
        --max-files "$MAX_FILES" \
        --audio-base-path "$AUDIO_BASE" \
        --output-base-path "/home/mku666/riffusion-hobby/results/spectrogram_images"
    
    if [ $? -eq 0 ]; then
        echo "  Successfully processed $TONE"
    else
        echo "  Failed to process $TONE"
    fi
    
    echo ""
done

echo "All tones processed!"
echo "Check results in: /home/mku666/riffusion-hobby/results/spectrogram_images/"
