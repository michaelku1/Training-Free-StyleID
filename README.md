# :guitar: Riffusion (hobby)


## riffusion
generate spectrograms --> riffusion/spectrogram_image_converter.py

riffusion server --> riffusion/server.py

generation --> curl -X POST http://127.0.0.1:8080/run_inference/ -H "Content-Type: application/json" -d '{"start":{"prompt":"","seed":42,"denoising":0.75,"guidance":7.0},"end":{"prompt":"","seed":123,"denoising":0.75,"guidance":7.0},"alpha":0.5,"num_inference_steps":200,"seed_image_id":"Chopper_egdb_1_spectrogram_image"}'

## stable audio
TBD

## style injection
TBD
