#!/bin/bash

# Generate spectrograms for instrument directories - works with three-level structure
# Usage: ./generate_spectrograms.sh [max_files_per_instrument] [instrument_path]
# Example: ./generate_spectrograms.sh 5 /home/mku666/riffusion-hobby/data_acoustic/musicTI/accordion

MAX_FILES=${1:-5}
INSTRUMENT_PATH=${2:-"/home/mku666/riffusion-hobby/data_acoustic/musicTI/accordion"}

echo "=== GENERATING SPECTROGRAMS FOR INSTRUMENT ==="
echo "Instrument Path: $INSTRUMENT_PATH"
echo "Max Files: $MAX_FILES"
echo ""

# Check if the instrument directory exists and contains wav files
if [ ! -d "$INSTRUMENT_PATH" ]; then
    echo "ERROR: Instrument directory does not exist: $INSTRUMENT_PATH"
    exit 1
fi

# Check for wav files (follow symbolic links)
WAV_COUNT=$(find -L "$INSTRUMENT_PATH" -name "*.wav" | wc -l)
if [ $WAV_COUNT -eq 0 ]; then
    echo "ERROR: No .wav files found in $INSTRUMENT_PATH"
    exit 1
fi

# Extract instrument name and parent directory
INSTRUMENT=$(basename "$INSTRUMENT_PATH")
PARENT_DIR=$(dirname "$INSTRUMENT_PATH")
echo "Detected instrument: $INSTRUMENT"
echo "Found $WAV_COUNT .wav files"
echo "Parent directory: $PARENT_DIR"
echo ""

# Process the instrument directory
TONE_DIR="$INSTRUMENT_PATH"
TONE="$INSTRUMENT"

echo "Processing instrument: $TONE"

# Count available files directly in the instrument directory (follow symbolic links)
AVAILABLE_FILES=$(find -L "$TONE_DIR" -name "*.wav" | wc -l)
if [ $AVAILABLE_FILES -eq 0 ]; then
    echo "  No .wav files found in $TONE_DIR"
    exit 1
fi

FILES_TO_PROCESS=$((AVAILABLE_FILES < MAX_FILES ? AVAILABLE_FILES : MAX_FILES))
echo "  Available files: $AVAILABLE_FILES, Processing: $FILES_TO_PROCESS"

# Create temporary directory structure for Python script compatibility
TEMP_BASE="/tmp/riffusion_temp_$$"
TEMP_TONE_DIR="$TEMP_BASE/$TONE"
TEMP_DI_DIR="$TEMP_TONE_DIR/files"

echo "  Creating temporary directory structure..."
mkdir -p "$TEMP_DI_DIR"

# Create numbered symbolic links for compatibility with Python script sorting
# Use numeric sorting to preserve original file order
counter=1
while IFS= read -r wav_file; do
    if [ -f "$wav_file" ]; then
        ln -sf "$wav_file" "$TEMP_DI_DIR/${counter}.wav"
        counter=$((counter + 1))
    fi
done < <(ls "$INSTRUMENT_PATH"/*.wav | sort -V)

# Run the spectrogram generation with temporary structure
python riffusion/spectrogram_image_converter_flexible.py \
    --source-type "Tone" \
    --di "files" \
    --tone "$TONE" \
    --max-files "$MAX_FILES" \
    --device "cpu" \
    --audio-base-path "$TEMP_BASE" \
    --output-base-path "/home/mku666/riffusion-hobby/results/spectrogram_images"

PYTHON_EXIT_CODE=$?

# Clean up temporary directory
echo "  Cleaning up temporary files..."
rm -rf "$TEMP_BASE"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "  Successfully processed $TONE"
else
    echo "  Failed to process $TONE"
    exit 1
fi

echo ""

echo "Instrument processing complete!"
echo "Check results in: /home/mku666/riffusion-hobby/results/spectrogram_images/"
