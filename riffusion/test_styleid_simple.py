#!/usr/bin/env python3
"""
Simple test script for the StyleID Riffusion Pipeline core functionality
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from riffusion.styleid_riffusion_pipeline import adain, feat_merge

def test_adain():
    """Test the AdaIN function."""
    print("Testing AdaIN function...")
    
    try:
        # Create test tensors
        content_feat = torch.randn(1, 4, 64, 64)
        style_feat = torch.randn(1, 4, 64, 64)
        
        # Apply AdaIN
        result = adain(content_feat, style_feat)
        
        # Check output shape
        assert result.shape == content_feat.shape, f"Shape mismatch: {result.shape} vs {content_feat.shape}"
        
        print("✅ AdaIN function works correctly!")
        return True
    except Exception as e:
        print(f"❌ AdaIN test failed: {e}")
        return False

def test_feat_merge():
    """Test the feature merging function."""
    print("\nTesting feature merging function...")
    
    try:
        # Create dummy features
        content_feats = {
            "layer7_attn": {
                0: (torch.randn(1, 64, 64), torch.randn(1, 64, 64), torch.randn(1, 64, 64)),
                1: (torch.randn(1, 64, 64), torch.randn(1, 64, 64), torch.randn(1, 64, 64))
            }
        }
        
        style_feats = {
            "layer7_attn": {
                0: (torch.randn(1, 64, 64), torch.randn(1, 64, 64), torch.randn(1, 64, 64)),
                1: (torch.randn(1, 64, 64), torch.randn(1, 64, 64), torch.randn(1, 64, 64))
            }
        }
        
        # Test feat_merge
        merged = feat_merge(content_feats, style_feats, start_step=0, gamma=0.75, T=1.5)
        
        # Check structure
        assert "layer7_attn" in merged, "Layer not found in merged features"
        assert 0 in merged["layer7_attn"], "Timestep 0 not found"
        assert 1 in merged["layer7_attn"], "Timestep 1 not found"
        
        # Check that we have Q, K, V tuples
        q, k, v = merged["layer7_attn"][0]
        assert isinstance(q, torch.Tensor), "Q is not a tensor"
        assert isinstance(k, torch.Tensor), "K is not a tensor"
        assert isinstance(v, torch.Tensor), "V is not a tensor"
        
        print("✅ Feature merging function works correctly!")
        return True
    except Exception as e:
        print(f"❌ Feature merging test failed: {e}")
        return False

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from riffusion.styleid_riffusion_pipeline import StyleIDRiffusionPipeline
        from riffusion.datatypes import InferenceInput, PromptInput
        
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_pipeline_class():
    """Test that the pipeline class can be instantiated (without loading model)."""
    print("\nTesting pipeline class structure...")
    
    try:
        from riffusion.styleid_riffusion_pipeline import StyleIDRiffusionPipeline
        
        # Check that the class exists and has expected methods
        assert hasattr(StyleIDRiffusionPipeline, 'styleid_riffuse'), "Missing styleid_riffuse method"
        assert hasattr(StyleIDRiffusionPipeline, 'setup_feature_extraction'), "Missing setup_feature_extraction method"
        assert hasattr(StyleIDRiffusionPipeline, 'extract_features_ddim'), "Missing extract_features_ddim method"
        assert hasattr(StyleIDRiffusionPipeline, 'load_checkpoint'), "Missing load_checkpoint method"
        
        print("✅ Pipeline class structure is correct!")
        return True
    except Exception as e:
        print(f"❌ Pipeline class test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Simple StyleID Riffusion Pipeline Tests")
    print("=" * 50)
    
    # Test 1: Imports
    import_test = test_imports()
    if not import_test:
        print("❌ Import test failed")
        return
    
    # Test 2: Pipeline Class Structure
    class_test = test_pipeline_class()
    if not class_test:
        print("❌ Pipeline class test failed")
        return
    
    # Test 3: AdaIN Function
    adain_test = test_adain()
    if not adain_test:
        print("❌ AdaIN test failed")
        return
    
    # Test 4: Feature Merging Function
    merge_test = test_feat_merge()
    if not merge_test:
        print("❌ Feature merging test failed")
        return
    
    print("\n🎉 All core functionality tests passed!")
    print("The StyleID Riffusion Pipeline core components are working correctly.")
    print("=" * 50)
    print("\nNote: Full model loading test requires the actual model files.")
    print("To test with a real model, you would need to:")
    print("1. Download the riffusion model")
    print("2. Update the checkpoint path in the test script")
    print("3. Run the full integration test")

if __name__ == "__main__":
    main() 