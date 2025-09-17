#!/bin/bash
while true; do
    USED=$(gpustat --json | jq '.gpus[1]["memory.used"]')
    if [ "$USED" -lt 10000 ]; then
        echo "Running job!"
        CUDA_VISIBLE_DEVICES=2 python -m riffusion.server --host 127.0.0.1 --port 8080
        break
    else
        echo "GPU 1 is currently full (used: $USED MB)"
    fi

    sleep 5
done