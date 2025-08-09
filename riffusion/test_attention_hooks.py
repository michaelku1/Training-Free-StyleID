#!/usr/bin/env python3
"""
Simple test for attention hooks without DDIM inversion.
"""

import torch
import numpy as np
from PIL import Image
import sys

# Add the riffusion directory to the path
sys.path.append('riffusion')

from styleid_riffusion_pipeline import StyleIDRiffusionPipeline

def test_attention_hooks():
    """Test attention hooks without DDIM inversion."""
    print("Testing Attention Hooks...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load the pipeline
        print("Loading StyleID Riffusion pipeline...")
        pipeline = StyleIDRiffusionPipeline.load_checkpoint(
            checkpoint="riffusion/riffusion-model-v1",
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=False
        )
        print("Pipeline loaded successfully!")
        
        # Test the feature extraction setup
        print("Testing feature extraction setup...")
        pipeline.setup_feature_extraction()
        print("Feature extraction setup completed!")
        
        # Test that hooks are registered
        if hasattr(pipeline, 'attention_hooks') and len(pipeline.attention_hooks) > 0:
            print(f"✓ {len(pipeline.attention_hooks)} attention hooks registered")
        else:
            print("✗ No attention hooks registered")
            return False
        
        # Test a simple forward pass to trigger the hooks
        print("Testing attention hooks with simple forward pass...")
        
        # Create a simple test input
        batch_size = 1
        channels = 4
        height = 64
        width = 64
        
        # Create test latents
        test_latents = torch.randn(batch_size, channels, height, width, device=device, dtype=pipeline.unet.dtype)
        
        # Create test timestep
        test_timestep = torch.tensor([100], device=device, dtype=torch.long)
        
        # Create dummy text embeddings
        hidden_size = pipeline.text_encoder.config.hidden_size
        dummy_embeddings = torch.zeros(batch_size, 77, hidden_size, device=device, dtype=pipeline.unet.dtype)
        
        # Enable feature extraction
        pipeline.trigger_get_qkv = True
        pipeline.trigger_modify_qkv = False
        pipeline.current_timestep = test_timestep.item()
        
        try:
            # Run a simple forward pass
            with torch.no_grad():
                output = pipeline.unet(test_latents, test_timestep, encoder_hidden_states=dummy_embeddings)
            
            print("✓ Forward pass completed successfully")
            
            # Check if features were extracted
            if hasattr(pipeline, 'attn_features') and pipeline.attn_features:
                print(f"✓ Features extracted for {len(pipeline.attn_features)} layers")
                for layer_name, layer_features in pipeline.attn_features.items():
                    print(f"  - {layer_name}: {len(layer_features)} timesteps")
                    for timestep, (q, k, v) in layer_features.items():
                        print(f"    Timestep {timestep}: Q{q.shape}, K{k.shape}, V{v.shape}")
            else:
                print("✗ No features extracted")
                return False
                
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("Testing cleanup...")
        # Test cleanup
        pipeline.cleanup_attention_hooks()
        print("✓ Attention hooks cleaned up")
        
        print("\n🎉 Attention hooks test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attention_hooks()
    if success:
        print("\n✅ Attention hooks test completed successfully!")
    else:
        print("\n❌ Attention hooks test failed!")
        sys.exit(1) 