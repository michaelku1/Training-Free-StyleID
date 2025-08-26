#!/usr/bin/env python3
"""
Script to decode JSON response from riffusion API and extract audio as WAV file.
"""

import json
import base64
import io
from pathlib import Path

def decode_audio_from_json(json_file_path, output_wav_path):
    """
    Decode audio from JSON response and save as WAV file.
    
    Args:
        json_file_path: Path to the JSON file containing the API response
        output_wav_path: Path where to save the WAV file
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the audio data (base64 encoded)
    audio_base64 = data.get('audio', '')
    
    if not audio_base64:
        print("No audio data found in JSON response")
        return
    
    # Remove the data URL prefix if present
    if audio_base64.startswith('data:audio/mpeg;base64,'):
        audio_base64 = audio_base64.replace('data:audio/mpeg;base64,', '')
    
    # Decode base64 to binary
    try:
        audio_binary = base64.b64decode(audio_base64)
        print(f"Successfully decoded {len(audio_binary)} bytes of audio data")
        
        # Save as WAV file
        with open(output_wav_path, 'wb') as f:
            f.write(audio_binary)
        
        print(f"Audio saved as: {output_wav_path}")
        
        # Also save the spectrogram image if present
        image_base64 = data.get('image', '')
        if image_base64:
            if image_base64.startswith('data:image/jpeg;base64,'):
                image_base64 = image_base64.replace('data:image/jpeg;base64,', '')
            
            image_binary = base64.b64decode(image_base64)
            image_path = output_wav_path.replace('.wav', '_spectrogram.jpg')
            
            with open(image_path, 'wb') as f:
                f.write(image_binary)
            
            print(f"Spectrogram saved as: {image_path}")
        
        # Print duration
        duration = data.get('duration_s', 0)
        print(f"Audio duration: {duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error decoding audio: {e}")

if __name__ == "__main__":
    # Decode the response.json file
    decode_audio_from_json('response.json', '/home/mku666/riffusion-hobby/results/riffusion_test/output_audio.wav') 