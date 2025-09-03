#!/bin/bash

# Batch script to run inference with EGDB_DI_1 chopper and clean image combinations
# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

# Base URL for the API
API_URL="http://127.0.0.1:8080/run_inference/"

# Function to run inference with given parameters
run_inference() {
    local seed_path="$1"
    local mask_path="$2"
    local description="$3"
    
    echo "Running inference: $description"
    echo "Seed image: $seed_path"
    echo "Mask image: $mask_path"
    echo "----------------------------------------"
    
    curl -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"start\": {
                \"prompt\": \"\",
                \"seed\": 42,
                \"denoising\": 0.2,
                \"guidance\": 0
            },
            \"num_inference_steps\": 50,
            \"seed_image_path\": \"$seed_path\",
            \"mask_image_path\": \"$mask_path\",
            \"alpha\": 0,
            \"end\": {
                \"prompt\": \"\",
                \"seed\": 123,
                \"denoising\": 0.2,
                \"guidance\": 0
            }
        }"
    
    echo -e "\n\n"
}

# Base path for images
BASE_PATH="/home/mku666/riffusion-hobby/results/riffusion_seed_mask_images/EGDB_DI_1"

echo "Starting batch inference runs with EGDB_DI_1 images..."
echo "=========================================="

# Run all combinations of clean vs chopper images
echo "Running all clean vs chopper combinations:"
for i in {1..10}; do
    for j in {1..10}; do
        seed_path="$BASE_PATH/clean/$i"
        mask_path="$BASE_PATH/chopper/$j"
        run_inference "$seed_path" "$mask_path" "clean$i vs chopper$j"
    done
done

# Run all combinations of chopper vs clean images (reverse)
echo "Running all chopper vs clean combinations:"
for i in {1..10}; do
    for j in {1..10}; do
        seed_path="$BASE_PATH/chopper/$i"
        mask_path="$BASE_PATH/clean/$j"
        run_inference "$seed_path" "$mask_path" "chopper$i vs clean$j"
    done
done

echo "=========================================="
echo "Batch inference completed!"
echo "Total combinations run: 200 (100 clean vs chopper + 100 chopper vs clean)"
