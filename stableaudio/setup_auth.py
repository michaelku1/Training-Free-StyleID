#!/usr/bin/env python3
"""
Helper script to set up Hugging Face authentication for Stable Audio Open.
"""

import subprocess
import sys
import os

def check_hf_login():
    """Check if user is logged in to Hugging Face."""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Already logged in as: {result.stdout.strip()}")
            return True
        else:
            print("✗ Not logged in to Hugging Face")
            return False
    except FileNotFoundError:
        print("✗ huggingface-cli not found. Make sure huggingface_hub is installed.")
        return False

def setup_auth():
    """Guide user through authentication setup."""
    print("🔐 Setting up Hugging Face Authentication for Stable Audio Open")
    print("=" * 60)
    
    # Check if already logged in
    if check_hf_login():
        print("\n✅ You're already authenticated!")
        return True
    
    print("\n📋 To access the Stable Audio Open model, you need to:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'read' permissions")
    print("3. Accept the model license at https://huggingface.co/stabilityai/stable-audio-open-1.0")
    print("4. Run the login command below")
    
    print("\n🚀 Running login command...")
    try:
        subprocess.run(['huggingface-cli', 'login'], check=True)
        print("\n✅ Login successful!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Login failed. Please try again manually:")
        print("   huggingface-cli login")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Login cancelled by user")
        return False

def test_model_access():
    """Test if we can access the model."""
    print("\n🧪 Testing model access...")
    try:
        from stable_audio_tools import get_pretrained_model
        model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        print("✅ Model access successful!")
        return True
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        print("\n💡 Make sure you:")
        print("   - Are logged in to Hugging Face")
        print("   - Have accepted the model license")
        print("   - Have a stable internet connection")
        return False

if __name__ == "__main__":
    print("🎵 Stable Audio Open Authentication Setup")
    print("=" * 40)
    
    if setup_auth():
        test_model_access()
    else:
        print("\n❌ Setup incomplete. Please try again.")
        sys.exit(1) 