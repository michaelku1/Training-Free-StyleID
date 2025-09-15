#!/bin/bash

# Batch script to run inference with user-specified DI directory and tone combinations
# Usage: ./batch_run.sh <DI_DIRECTORY> <NUM_FILES>
# Example: ./batch_run.sh DI_1 10

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <DI_DIRECTORY> <NUM_FILES>"
    echo "Example: $0 DI_1 10"
    echo "Available DI directories:"
    ls -1 /home/mku666/riffusion-hobby/sample_data/fx_data/EGDB-Large-Subset/AudioDI/ | grep "^DI_" | sort -V
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

# Get user input parameters
DI_DIRECTORY="$1"
NUM_FILES="$2"

# Base URL for the API
API_URL="http://127.0.0.1:8080/run_inference/"

# Ensure output directory exists (used by server to write JSON)
OUTPUT_DIR="/home/mku666/riffusion-hobby/results/audio"
mkdir -p "$OUTPUT_DIR"

# Base paths
AUDIO_DI_BASE="/home/mku666/riffusion-hobby/sample_data/fx_data/EGDB-Large-Subset/AudioDI"
TONE_BASE="/home/mku666/riffusion-hobby/sample_data/fx_data/EGDB-Large-Subset/Tone"

# Validate DI directory exists
if [ ! -d "$AUDIO_DI_BASE/$DI_DIRECTORY" ]; then
    echo "Error: DI directory '$DI_DIRECTORY' not found in $AUDIO_DI_BASE"
    echo "Available directories:"
    ls -1 "$AUDIO_DI_BASE" | grep "^DI_" | sort -V
    exit 1
fi

# Get list of wav files from the specified DI directory, sorted numerically
WAV_FILES=($(ls "$AUDIO_DI_BASE/$DI_DIRECTORY"/*.wav | sort -V | head -n "$NUM_FILES"))

# Check if we have enough files
if [ ${#WAV_FILES[@]} -lt "$NUM_FILES" ]; then
    echo "Warning: Only found ${#WAV_FILES[@]} wav files in $DI_DIRECTORY, requested $NUM_FILES"
    NUM_FILES=${#WAV_FILES[@]}
fi

# List of tone names from EGDB-Large-Subset/Tone directory
TONE_NAMES=(
    "Moore Clean"
    "Rhapsody"
    "First Compression"
    "New Guitar Icon"
    "Gravity"
    "Chopper"
    "Room 808"
    "Light House"
    "Dark Soul"
    "Easy Blues"
)

# Function to run inference with given parameters
run_inference() {
    local seed_path="$1"
    local mask_path="$2"
    local description="$3"
    
    echo "Running inference: $description"
    echo "Seed audio: $seed_path"
    echo "Mask tone: $mask_path"
    echo "----------------------------------------"
    
    curl --fail --silent --show-error -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"start\": {
                \"prompt\": \"\",
                \"seed\": 42,
                \"denoising\": 0.2,
                \"guidance\": 0
            },
            \"end\": {
                \"prompt\": \"\",
                \"seed\": 123,
                \"denoising\": 0.2,
                \"guidance\": 0
            },
            \"alpha\": 0.0,
            \"num_inference_steps\": 50,
            \"seed_image_path\": \"$seed_path\",
            \"mask_image_path\": \"$mask_path\",
            \"output_path\": \"$OUTPUT_DIR\"
        }" || echo "Request failed for: $description"
    
    echo -e "\n\n"
}

echo "Starting batch inference runs..."
echo "DI Directory: $DI_DIRECTORY"
echo "Number of files: $NUM_FILES"
echo "WAV files to use:"
for i in "${!WAV_FILES[@]}"; do
    echo "  $((i+1)). $(basename "${WAV_FILES[$i]}")"
done
echo "=========================================="

# Run all combinations of DI files vs tone images
echo "Running all DI files vs tone combinations:"
for i in "${!WAV_FILES[@]}"; do
    for tone_name in "${TONE_NAMES[@]}"; do
        for j in {1..10}; do
            seed_path="${WAV_FILES[$i]}"
            mask_path="$TONE_BASE/$tone_name/DI_$j"
            file_num=$((i+1))
            run_inference "$seed_path" "$mask_path" "DI_${file_num}_$(basename "$seed_path") vs $tone_name$j"
        done
    done
done

echo "=========================================="
echo "Batch inference completed!"
echo "Total combinations run: $((NUM_FILES * ${#TONE_NAMES[@]} * 10)) (DI files vs all tones)"
