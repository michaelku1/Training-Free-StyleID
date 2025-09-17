json_payload='{"start":{"prompt":"A jazzy piano solo","seed":42,"denoising":0.75,"guidance":7.0},"end":{"prompt":"A smooth saxophone melody","seed":123,"denoising":0.75,"guidance":7.0},"alpha":0.5,"num_inference_steps":50,"seed_image_id":"og_beat"}'

curl -X POST http://127.0.0.1:8080/run_inference/ \
  -H "Content-Type: application/json" \
  -d "$json_payload"

