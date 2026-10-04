#!/usr/bin/env bash
# Serve the teacher model with one OpenAI-compatible vLLM server per GPU.
# List the resulting URLs (http://HOST:PORT/v1) under teacher.base_urls in configs/teacher.yaml.
set -euo pipefail
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the teacher checkpoint}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.6-27B}
NUM_GPUS=${NUM_GPUS:-8}
BASE_PORT=${BASE_PORT:-8100}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
LOG_DIR=${LOG_DIR:-logs/teacher_vllm}
mkdir -p "$LOG_DIR"

pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    port=$((BASE_PORT + gpu))
    echo "GPU $gpu -> http://0.0.0.0:$port/v1"
    CUDA_VISIBLE_DEVICES=$gpu "${PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
        --host 0.0.0.0 --port "$port" --tensor-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" >"$LOG_DIR/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
wait
