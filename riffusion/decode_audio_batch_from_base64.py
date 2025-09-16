import json
import base64
import io
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_audio_from_json(json_file_path, output_wav_path):
    """
    Decode audio from JSON response and save as WAV file.

    Args:
        json_file_path: Path to the JSON file containing the API response
        output_wav_path: Path where to save the WAV file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Extract the audio data (base64 encoded)
        audio_base64 = data.get('audio', '')

        if not audio_base64:
            logger.warning(f"No audio data found in {json_file_path}")
            return False

        # Remove the data URL prefix if present
        if audio_base64.startswith('data:audio/mpeg;base64,'):
            audio_base64 = audio_base64.replace('data:audio/mpeg;base64,', '')
        elif audio_base64.startswith('data:audio/wav;base64,'):
            audio_base64 = audio_base64.replace('data:audio/wav;base64,', '')

        # Decode base64 to binary
        audio_binary = base64.b64decode(audio_base64)
        logger.info(f"Successfully decoded {len(audio_binary)} bytes of audio data from {json_file_path}")

        # Create output directory if it doesn't exist
        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as WAV file
        with open(output_wav_path, 'wb') as f:
            f.write(audio_binary)

        logger.info(f"Audio saved as: {output_wav_path}")

        # Also save the spectrogram image if present (optional)
        image_base64 = data.get('image', '')
        if image_base64:
            if image_base64.startswith('data:image/jpeg;base64,'):
                image_base64 = image_base64.replace('data:image/jpeg;base64,', '')
            elif image_base64.startswith('data:image/png;base64,'):
                image_base64 = image_base64.replace('data:image/png;base64,', '')

            image_binary = base64.b64decode(image_base64)

            # Uncomment if you want to save spectrograms
            # image_path = output_wav_path.replace('.wav', '_spectrogram.jpg')
            # with open(image_path, 'wb') as f:
            #     f.write(image_binary)
            # logger.info(f"Spectrogram saved as: {image_path}")

        # Print duration if available
        duration = data.get('duration_s', 0)
        if duration > 0:
            logger.info(f"Audio duration: {duration:.2f} seconds")

        return True

    except json.JSONDecodeError as e:
        logger.error(f"Error reading JSON file {json_file_path}: {e}")
        return False
    except base64.binascii.Error as e:
        logger.error(f"Error decoding base64 data from {json_file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing {json_file_path}: {e}")
        return False

def batch_decode_audio_from_directory(directory_path, recursive=True):
    """
    Process all JSON files in a directory (and optionally its subdirectories) and decode audio from each one.

    Args:
        directory_path: Path to the directory containing JSON files
        recursive: If True, search subdirectories recursively (default: True)
    """
    directory = Path(directory_path)

    if not directory.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return

    if not directory.is_dir():
        logger.error(f"Path is not a directory: {directory_path}")
        return

    # Find all JSON files in the directory (and subdirectories if recursive)
    if recursive:
        json_files = list(directory.rglob("*.json"))  # rglob for recursive search
        logger.info(f"Recursively searching for JSON files in: {directory_path}")
    else:
        json_files = list(directory.glob("*.json"))   # glob for current directory only
        logger.info(f"Searching for JSON files in: {directory_path}")

    if not json_files:
        logger.warning(f"No JSON files found in directory: {directory_path}")
        return

    logger.info(f"Found {len(json_files)} JSON files to process")

    # Group files by subdirectory for better logging
    subdirs = {}
    for json_file in json_files:
        relative_path = json_file.relative_to(directory)
        subdir = str(relative_path.parent) if relative_path.parent != Path('.') else 'root'
        if subdir not in subdirs:
            subdirs[subdir] = []
        subdirs[subdir].append(json_file)

    # Log the distribution of files across subdirectories
    for subdir, files in subdirs.items():
        logger.info(f"  {subdir}: {len(files)} JSON file(s)")

    successful_conversions = 0
    failed_conversions = 0

    # Process each JSON file
    for json_file in json_files:
        # Get relative path for better logging
        relative_path = json_file.relative_to(directory)
        logger.info(f"Processing: {relative_path}")

        # Generate output WAV path (same location as JSON file, but with .wav extension)
        wav_file = json_file.with_suffix('.wav')

        # Skip if WAV file already exists (optional - remove this check if you want to overwrite)
        if wav_file.exists():
            logger.info(f"WAV file already exists, skipping: {wav_file.relative_to(directory)}")
            continue

        # Decode the audio
        success = decode_audio_from_json(json_file, wav_file)

        if success:
            successful_conversions += 1
        else:
            failed_conversions += 1

        logger.info("-" * 50)  # Separator for readability

    # Summary
    logger.info(f"Batch processing completed!")
    logger.info(f"Successfully converted: {successful_conversions} files")
    logger.info(f"Failed conversions: {failed_conversions} files")
    logger.info(f"Total processed: {len(json_files)} files")

def batch_decode_audio_from_directory_non_recursive(directory_path):
    """
    Convenience function to process only the specified directory without recursion.

    Args:
        directory_path: Path to the directory containing JSON files
    """
    batch_decode_audio_from_directory(directory_path, recursive=False)


# Process all JSON files recursively in subdirectories
DIR_PATH = "/home/mku666/riffusion-hobby/results/audio/"
batch_decode_audio_from_directory(DIR_PATH)