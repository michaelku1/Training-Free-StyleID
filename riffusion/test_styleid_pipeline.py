#!/usr/bin/env python3
"""
Test script for the StyleID Riffusion Pipeline
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from riffusion.styleid_riffusion_pipeline import StyleIDRiffusionPipeline
from riffusion.datatypes import InferenceInput, PromptInput

def create_test_images():
    """Create simple test images for content and style."""
    # Create a simple content image (spectrogram-like)
    content_array = np.random.rand(512, 512, 3) * 255
    content_image = Image.fromarray(content_array.astype(np.uint8))
    
    # Create a simple style image (different pattern)
    style_array = np.random.rand(512, 512, 3) * 255
    style_image = Image.fromarray(style_array.astype(np.uint8))
    
    return content_image, style_image

def create_test_inputs():
    """Create test inference inputs."""
    start_params = PromptInput(
        prompt="electronic music",
        seed=42,
        guidance=7.5,
        denoising=0.8
    )
    
    end_params = PromptInput(
        prompt="jazz music", 
        seed=123,
        guidance=7.5,
        denoising=0.8
    )
    
    inputs = InferenceInput(
        alpha=0.5,  # Interpolate halfway between start and end
        start=start_params,
        end=end_params,
        num_inference_steps=20  # Use fewer steps for testing
    )
    
    return inputs

def test_pipeline_loading():
    """Test if the pipeline can be loaded successfully."""
    print("Testing pipeline loading...")
    
    try:
        # Try to load the pipeline
        pipeline = StyleIDRiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            device="cpu",  # Use CPU for testing
            dtype=torch.float32,
            local_files_only=False
        )
        print("✅ Pipeline loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return None

def test_feature_extraction(pipeline):
    """Test feature extraction functionality."""
    print("\nTesting feature extraction...")
    
    try:
        # Create test images
        content_image, style_image = create_test_images()
        
        # Test setup_feature_extraction
        pipeline.setup_feature_extraction()
        print("✅ Feature extraction setup successful!")
        
        # Test cleanup
        pipeline.cleanup_attention_hooks()
        print("✅ Hook cleanup successful!")
        
        return True
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_styleid_inference(pipeline):
    """Test the full StyleID inference process."""
    print("\nTesting StyleID inference...")
    
    try:
        # Create test inputs
        inputs = create_test_inputs()
        content_image, style_image = create_test_images()
        
        # Run StyleID inference
        print("Running StyleID inference...")
        result_image = pipeline.styleid_riffuse(
            inputs=inputs,
            content_image=content_image,
            style_image=style_image,
            use_adain_init=True,
            use_attn_injection=True,
            gamma=0.75,
            T=1.5,
            start_step=10  # Start injection earlier for testing
        )
        
        print("✅ StyleID inference completed successfully!")
        
        # Save the result
        result_image.save("test_styleid_result.png")
        print("✅ Result saved as test_styleid_result.png")
        
        return True
    except Exception as e:
        print(f"❌ StyleID inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Starting StyleID Riffusion Pipeline Tests")
    print("=" * 50)
    
    # Test 1: Pipeline Loading
    pipeline = test_pipeline_loading()
    if pipeline is None:
        print("❌ Cannot continue without pipeline")
        return
    
    # Test 2: Feature Extraction
    feature_test = test_feature_extraction(pipeline)
    if not feature_test:
        print("❌ Feature extraction test failed")
        return
    
    # Test 3: Full StyleID Inference
    inference_test = test_styleid_inference(pipeline)
    if not inference_test:
        print("❌ StyleID inference test failed")
        return
    
    print("\n🎉 All tests passed! The StyleID Riffusion Pipeline is working correctly.")
    print("=" * 50)

if __name__ == "__main__":
    main() 