#!/bin/bash

# Batch script to run inference with spectrogram images
# Usage: ./batch_run_spectrogram.sh <SEED_DIRECTORY> <MASK_DIRECTORY> <NUM_SEED_IMAGES>
# Example: ./batch_run_spectrogram.sh AudioDI_DI_1 "Tone_Moore Clean_DI_1" 5

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <SEED_DIRECTORY> <MASK_DIRECTORY> <NUM_SEED_IMAGES>"
    echo "Example: $0 AudioDI_DI_1 \"Tone_Moore Clean_DI_1\" 5"
    echo ""
    echo "Available directories:"
    echo "Seed directories (AudioDI):"
    ls -1 /home/mku666/riffusion-hobby/results/spectrogram_images/ | grep "^AudioDI_" | sort -V
    echo ""
    echo "Mask directories (Tone):"
    ls -1 /home/mku666/riffusion-hobby/results/spectrogram_images/ | grep "^Tone_" | sort -V
    exit 1
fi

# Set CUDA device (use environment variable if set, otherwise default to 1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Get user input parameters
SEED_DIRECTORY="$1"
MASK_DIRECTORY="$2"
NUM_SEED_IMAGES="$3"

# Base URL for the API
API_URL="http://127.0.0.1:8080/run_inference/"

# Ensure output directory exists (used by server to write JSON)
OUTPUT_DIR="/home/mku666/riffusion-hobby/results/audio"
mkdir -p "$OUTPUT_DIR"

# Base path for spectrogram images
SPECTROGRAM_BASE="/home/mku666/riffusion-hobby/results/spectrogram_images"

# Validate directories exist
if [ ! -d "$SPECTROGRAM_BASE/$SEED_DIRECTORY" ]; then
    echo "Error: Seed directory '$SEED_DIRECTORY' not found in $SPECTROGRAM_BASE"
    echo "Available seed directories:"
    ls -1 "$SPECTROGRAM_BASE" | grep "^AudioDI_" | sort -V
    exit 1
fi

if [ ! -d "$SPECTROGRAM_BASE/$MASK_DIRECTORY" ]; then
    echo "Error: Mask directory '$MASK_DIRECTORY' not found in $SPECTROGRAM_BASE"
    echo "Available mask directories:"
    ls -1 "$SPECTROGRAM_BASE" | grep "^Tone_" | sort -V
    exit 1
fi

# Get list of PNG files from the seed directory, sorted numerically
SEED_IMAGES=($(ls "$SPECTROGRAM_BASE/$SEED_DIRECTORY"/*.png | sort -V | head -n "$NUM_SEED_IMAGES"))

# Check if we have enough seed images
if [ ${#SEED_IMAGES[@]} -lt "$NUM_SEED_IMAGES" ]; then
    echo "Warning: Only found ${#SEED_IMAGES[@]} PNG files in $SEED_DIRECTORY, requested $NUM_SEED_IMAGES"
    NUM_SEED_IMAGES=${#SEED_IMAGES[@]}
fi

# Get all PNG files from the mask directory
MASK_IMAGES=($(ls "$SPECTROGRAM_BASE/$MASK_DIRECTORY"/*.png | sort -V))

# Check if we have mask images
if [ ${#MASK_IMAGES[@]} -eq 0 ]; then
    echo "Error: No PNG files found in mask directory '$MASK_DIRECTORY'"
    exit 1
fi

# Function to run inference with given parameters
run_inference() {
    local seed_path="$1"
    local mask_path="$2"
    local description="$3"
    
    echo "Running inference: $description"
    echo "Seed image: $seed_path"
    echo "Mask image: $mask_path"
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
echo "Seed Directory: $SEED_DIRECTORY"
echo "Mask Directory: $MASK_DIRECTORY"
echo "Number of seed images: $NUM_SEED_IMAGES"
echo "Number of mask images: ${#MASK_IMAGES[@]}"
echo "Seed images to use:"
for i in "${!SEED_IMAGES[@]}"; do
    echo "  $((i+1)). $(basename "${SEED_IMAGES[$i]}")"
done
echo "=========================================="

# Run all combinations of seed images vs mask images
echo "Running all seed vs mask combinations:"
for i in "${!SEED_IMAGES[@]}"; do
    for j in "${!MASK_IMAGES[@]}"; do
        seed_path="${SEED_IMAGES[$i]%.png}"  # Remove .png extension
        mask_path="${MASK_IMAGES[$j]%.png}"  # Remove .png extension
        seed_num=$((i+1))
        mask_num=$((j+1))
        run_inference "$seed_path" "$mask_path" "Seed_${seed_num}_$(basename "$seed_path") vs Mask_${mask_num}_$(basename "$mask_path")"
    done
done

echo "=========================================="
echo "Batch inference completed!"
echo "Total combinations run: $((NUM_SEED_IMAGES * ${#MASK_IMAGES[@]})) (seed images vs mask images)"
