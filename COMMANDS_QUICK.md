# Quick Commands Reference

## 🚀 Start Server
```bash
CUDA_VISIBLE_DEVICES=1 python -m riffusion.server --host 127.0.0.1 --port 8080
```

## 🎵 Basic Inference
```bash
# Simple
curl -X POST http://127.0.0.1:8080/run_inference/ -H "Content-Type: application/json" -d '{"start":{"prompt":"","seed":42,"denoising":0.75,"guidance":7.0},"num_inference_steps":50,"seed_image_id":"mask_beat_lines_80", "output_path": ""}'

# With mask
CUDA_VISIBLE_DEVICES=1 curl -X POST http://127.0.0.1:8080/run_inference/ -H "Content-Type: application/json" -d '{"start":{"prompt":"","seed":42,"denoising":0.2,"guidance":0},"num_inference_steps":50,"seed_image_path":"results/riffusion_seed_mask_images/accordian123/1.png","mask_image_path":"results/riffusion_seed_mask_images/violin123/1.png","alpha":0,"end":{"prompt":"","seed":123,"denoising":0.2,"guidance":0}}'
```

## 🎨 Style Transfer
```bash
# Riffusion StyleID
CUDA_VISIBLE_DEVICES=1 python3 styleid_inference.py --content_audio /mnt/gestalt/home/mku666/musicTI_audios/content/piano/piano1.wav --style_audio /mnt/gestalt/home/mku666/musicTI_audios/timbre/accordion/accordion1.wav --output_path ./piano_to_accordion_style_output.wav --prompt_start "" --prompt_end "" --start_step 200 --no_adain_init --attention_op_type 1

# Stable Audio
CUDA_VISIBLE_DEVICES=1 python styleid_inference_stableaudio.py --content_audio /mnt/gestalt/home/mku666/musicTI_audios/content/piano/piano1.wav --style_audio /mnt/gestalt/home/mku666/musicTI_audios/timbre/accordion/accordion1.wav --output_path ./piano_to_accordion_style_output.wav --gamma 0.8 --T 1.2 --start_step 45 --prompt "" --num_inference_steps 100
```

## 📊 Visualization
```bash
# Attention heatmap
python riffusion_activation_visualization.py --audio_path /mnt/gestalt/home/mku666/musicTI_audios/content/piano/piano1.wav --time_step 20 --attention_layer "6,7,8,9,10,11" --head 0 --prompt "electronic beats" --num_inference_steps 30 --seed 42

# Spectrogram plot
python3 spectrogram_plot.py --style_path sample_data/fx_data/EGDB-Large-Subset/Tone/Chopper/DI_1/1.wav --content_path sample_data/fx_data/EGDB-Large-Subset/AudioDI/DI_1/2.wav --output_path results/audio_w_reverse_mask/clean2_to_chopper1.wav --save_path results/audio_w_reverse_mask/reverse_mask_clean2_to_chopper1.png
```

## 🔧 Utils
```bash
# Decode audio
python3 decode_audio_from_base64.py --json_file_path results/audio/clean2_to_chopper1.json --output_wav_path ../results/audio/clean2_to_chopper1.wav
```

## 📝 Quick Notes
- GPU: `CUDA_VISIBLE_DEVICES=0` or `1`
- Params: `denoising` (0.2-0.75), `guidance` (0-7), `steps` (30-100)
- Outputs: `results/audio/`
