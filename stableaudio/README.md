# Stable Audio Open Testing

This directory contains test scripts for the Stable Audio Open model.

## Setup

### 1. Virtual Environment
The project uses a virtual environment located at `../venv/`. Make sure it's activated:

```bash
source ../venv/bin/activate
```

### 2. Dependencies
All required packages are already installed:
- `torch` and `torchaudio`
- `einops`
- `stable-audio-tools`
- `encodec`
- `k-diffusion`

### 3. Hugging Face Authentication
To access the Stable Audio Open model, you need to authenticate with Hugging Face:

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "read" permissions
3. Run the login command:
   ```bash
   huggingface-cli login
   ```
4. Enter your token when prompted

### 4. IDE Configuration
If you're using VS Code, the `.vscode/settings.json` file is configured to use the correct Python interpreter.

## Test Scripts

### `test_imports.py`
Tests all imports without downloading models:
```bash
python test_imports.py
```

### `test_stable_audio_open_download.py`
Tests the full model download and loading:
```bash
python test_stable_audio_open_download.py
```

## Troubleshooting

### Import Errors in IDE
If your IDE shows import errors:
1. Make sure you're using the correct Python interpreter (`../venv/bin/python`)
2. Restart your IDE after activating the virtual environment
3. Check that the `.vscode/settings.json` file is properly configured

### Model Access Errors
If you get 403 Forbidden errors:
1. Make sure you're logged in to Hugging Face
2. Check that your token has the correct permissions
3. The model requires accepting the license terms on the Hugging Face website

### CUDA Issues
If you have CUDA issues:
1. Check that PyTorch is installed with CUDA support
2. Verify your GPU drivers are up to date
3. The scripts will fall back to CPU if CUDA is not available 