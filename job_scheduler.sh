#!/bin/bash
while true; do
    FREE=$(gpustat --json | jq '.gpus[0].memory.free')
    if [ "$FREE" -gt 1000 ]; then
        echo "Running job!"
        ./your_script.sh
        break
    fi
    sleep 5
done