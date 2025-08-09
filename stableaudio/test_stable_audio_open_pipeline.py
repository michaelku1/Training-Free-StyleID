#!/usr/bin/env python3
"""
Test script for StyleID Stable Audio Open v1.0 pipeline.
This script tests the basic functionality of the updated pipeline.
"""

import torch
import numpy as np
from stable_audio_pipeline import StyleIDStableAudioOpenPipeline


def test_pipeline_loading():
    """Test that the pipeline can be loaded successfully."""
    try:
        # Test loading with placeholder components (when stable_audio_open is not available)
        pipeline = StyleIDStableAudioOpenPipeline.load_checkpoint(
            checkpoint="test_checkpoint",
            device="cpu",  # Use CPU for testing
            dtype=torch.float32
        )
        print("✓ Pipeline loaded successfully")
        return pipeline
    except Exception as e:
        print(f"✗ Failed to load pipeline: {e}")
        return None


def test_feature_extraction(pipeline):
    """Test feature extraction functionality."""
    try:
        # Create dummy audio data
        dummy_audio = torch.randn(1, 2, 44100)  # 1 second of stereo audio
        
        # Test feature extraction
        latents, features = pipeline.extract_features_ddim(
            dummy_audio, 
            num_steps=10,  # Use fewer steps for testing
            save_feature_steps=10
        )
        
        print(f"✓ Feature extraction successful")
        print(f"  - Latents shape: {latents.shape}")
        print(f"  - Number of features: {len(features)}")
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False


def test_styleid_generation(pipeline):
    """Test StyleID generation functionality."""
    try:
        # Create dummy inputs
        batch_size = 1
        text_embeddings = torch.randn(batch_size, 77, 768)
        init_latents = torch.randn(batch_size, 4, 64, 64)  # Dummy latent shape
        
        # Test generation
        outputs = pipeline.styleid_generate(
            text_embeddings=text_embeddings,
            init_latents=init_latents,
            num_inference_steps=10,  # Use fewer steps for testing
            guidance_scale=7.5
        )
        
        print(f"✓ StyleID generation successful")
        print(f"  - Output latents shape: {outputs['latents'].shape}")
        return True
    except Exception as e:
        print(f"✗ StyleID generation failed: {e}")
        return False


def test_diffusion_transformer_with_styleid(pipeline):
    """Test the DiffusionTransformer with StyleID injection."""
    try:
        # Create dummy inputs
        x = torch.randn(1, 4, 64, 64)
        t = torch.tensor([50])
        encoder_hidden_states = torch.randn(1, 77, 768)
        
        # Test forward pass
        output = pipeline.diffusion_transformer_with_styleid(
            x, t, encoder_hidden_states, injected_features=None
        )
        
        print(f"✓ DiffusionTransformer with StyleID successful")
        print(f"  - Output shape: {output.sample.shape}")
        return True
    except Exception as e:
        print(f"✗ DiffusionTransformer with StyleID failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing StyleID Stable Audio Open v1.0 Pipeline")
    print("=" * 50)
    
    # Test 1: Pipeline loading
    pipeline = test_pipeline_loading()
    if pipeline is None:
        print("Pipeline loading failed, skipping other tests")
        return
    
    # Test 2: Feature extraction
    test_feature_extraction(pipeline)
    
    # Test 3: StyleID generation
    test_styleid_generation(pipeline)
    
    # Test 4: DiffusionTransformer with StyleID
    test_diffusion_transformer_with_styleid(pipeline)
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main() 