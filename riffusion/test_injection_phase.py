#!/usr/bin/env python3
"""
Test script to verify that style injection happens during the DDIM reverse process.
"""

import sys
import os
import copy
import torch
from PIL import Image
import numpy as np

# Add riffusion to path
sys.path.append('riffusion')

try:
    from styleid_riffusion_pipeline import StyleIDRiffusionPipeline
    from datatypes import InferenceInput, PromptInput
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def create_test_spectrogram():
    """Create a simple test spectrogram."""
    # Create a simple gradient spectrogram
    height, width = 256, 256
    spectrogram = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient pattern
    for i in range(height):
        for j in range(width):
            spectrogram[i, j] = [
                int(255 * i / height),  # Red gradient
                int(255 * j / width),   # Green gradient
                128                      # Blue constant
            ]
    
    return Image.fromarray(spectrogram)

def test_injection_phase():
    """Test that style injection happens during DDIM reverse process."""
    print("Testing Style Injection During DDIM Reverse Process...")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pipeline
    print("Loading StyleID Riffusion pipeline...")
    try:
        pipeline = StyleIDRiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            device=device
        )
        print("Pipeline loaded successfully!")
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return False
    
    # Create test spectrograms
    print("Creating test spectrograms...")
    content_spectrogram = create_test_spectrogram()
    style_spectrogram = create_test_spectrogram()
    
    # Create test inputs
    inputs = InferenceInput(
        alpha=0.5,
        start=PromptInput(
            prompt="test prompt",
            seed=42,
            guidance=7.5,
            denoising=0.8
        ),
        end=PromptInput(
            prompt="test prompt",
            seed=43,
            guidance=7.5,
            denoising=0.8
        ),
        num_inference_steps=5  # Use small number for testing
    )
    
    # Test the full pipeline with injection tracking
    print("Testing full pipeline with injection tracking...")
    
    # Override the unet_with_styleid method to track injection
    original_unet_with_styleid = pipeline.unet_with_styleid
    injection_called = False
    injection_features = None
    
    def tracked_unet_with_styleid(x, t, encoder_hidden_states, injected_features=None):
        nonlocal injection_called, injection_features
        if injected_features is not None:
            injection_called = True
            injection_features = injected_features
            print(f"✓ Style injection called at timestep {t.item()}")
            print(f"  - Injected features for layers: {list(injected_features.keys())}")
        return original_unet_with_styleid(x, t, encoder_hidden_states, injected_features)
    
    pipeline.unet_with_styleid = tracked_unet_with_styleid
    
    try:
        # Run the full pipeline
        result = pipeline.styleid_riffuse(
            inputs=inputs,
            content_image=content_spectrogram,
            style_image=style_spectrogram,
            use_adain_init=True,
            use_attn_injection=True,
            gamma=0.75,
            T=1.5,
            start_step=0  # Start injection from beginning for testing
        )
        
        print("✓ Pipeline completed successfully")
        print(f"✓ Result image shape: {result.size}")
        
        if injection_called:
            print("✓ Style injection was called during DDIM reverse process")
            print(f"✓ Injection features: {injection_features}")
            return True
        else:
            print("✗ Style injection was NOT called during DDIM reverse process")
            return False
            
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = test_injection_phase()
    if success:
        print("\n🎉 Style injection test completed successfully!")
        print("✅ Style injection is happening during DDIM reverse process")
    else:
        print("\n❌ Style injection test failed!")
        print("❌ Style injection is NOT happening during DDIM reverse process")
    
    sys.exit(0 if success else 1) 