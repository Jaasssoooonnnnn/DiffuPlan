#!/bin/bash
set -euo pipefail

# Launch the retriever server. GPU index sharding is the default; set RETRIEVER_USE_GPU=0 for CPU retrieval.

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

FILE_PATH=${RETRIEVER_ASSET_DIR:-"$SCRIPT_DIR/../data/retrieval"}
INDEX_FILE=${INDEX_FILE:-"$FILE_PATH/e5_Flat.index"}
CORPUS_FILE=${CORPUS_FILE:-"$FILE_PATH/wiki-18.jsonl"}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_PATH=${RETRIEVER_PATH:-intfloat/e5-base-v2}

# Runtime knobs (overridable via env)
RETRIEVER_USE_GPU=${RETRIEVER_USE_GPU:-1}
RETRIEVER_GPU_IDS=${RETRIEVER_GPU_IDS:-}
RETRIEVER_GPU_SHARD=${RETRIEVER_GPU_SHARD:-1}
RETRIEVER_GPU_LOAD_MODE=${RETRIEVER_GPU_LOAD_MODE:-batch}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}
RETRIEVER_FORCE_CPU=${RETRIEVER_FORCE_CPU:-0}
RETRIEVER_PORT=${RETRIEVER_PORT:-8000}

is_truthy() {
    local value="${1:-}"
    shopt -s nocasematch
    if [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]; then
        shopt -u nocasematch
        return 0
    fi
    shopt -u nocasematch
    return 1
}

detect_visible_gpu_ids() {
    local count=0
    local ids=""

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local cvd="${CUDA_VISIBLE_DEVICES// /}"
        if [[ "$cvd" != "-1" && "$cvd" != "NoDevFiles" ]]; then
            IFS=',' read -r -a gpu_entries <<< "$cvd"
            for entry in "${gpu_entries[@]}"; do
                if [[ -n "$entry" ]]; then
                    count=$((count + 1))
                fi
            done
        fi
    fi

    if (( count == 0 )) && command -v nvidia-smi >/dev/null 2>&1; then
        count=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU [0-9]\+:')
    fi

    if (( count == 0 )); then
        echo ""
        return 0
    fi

    for ((i = 0; i < count; i++)); do
        if [[ -z "$ids" ]]; then
            ids="$i"
        else
            ids="$ids,$i"
        fi
    done
    echo "$ids"
}

# Check that the index file exists.
if [ ! -f "$INDEX_FILE" ]; then
    echo "Error: Index file not found at $INDEX_FILE"
    echo "Please ensure the index is properly set up."
    exit 1
fi

# Check that the corpus file exists.
if [ ! -f "$CORPUS_FILE" ]; then
    echo "Error: Corpus file not found at $CORPUS_FILE"
    exit 1
fi

USE_GPU=1
if is_truthy "$RETRIEVER_FORCE_CPU" || ! is_truthy "$RETRIEVER_USE_GPU"; then
    USE_GPU=0
fi

if [[ "$USE_GPU" == "1" ]] && [[ -z "$RETRIEVER_GPU_IDS" ]]; then
    RETRIEVER_GPU_IDS="$(detect_visible_gpu_ids)"
fi

if [[ "$USE_GPU" == "1" ]] && [[ -z "$RETRIEVER_GPU_IDS" ]]; then
    echo "[Warn] Unable to infer visible GPUs. Falling back to CPU retriever."
    USE_GPU=0
fi

GPU_SHARD_FLAG="--gpu_shard"
if ! is_truthy "$RETRIEVER_GPU_SHARD"; then
    GPU_SHARD_FLAG="--no_gpu_shard"
fi

MODE_LABEL="CPU"
if [[ "$USE_GPU" == "1" ]]; then
    MODE_LABEL="GPU"
fi

echo "=========================================="
echo "  Search-LLaDA2 Retriever Server (${MODE_LABEL})"
echo "=========================================="
echo "Index: $INDEX_FILE"
echo "Corpus: $CORPUS_FILE"
echo "Retriever: $RETRIEVER_NAME ($RETRIEVER_PATH)"
echo "TopK: $RETRIEVER_TOPK"
echo "Mode: $MODE_LABEL"
echo "Force CPU: $RETRIEVER_FORCE_CPU"
echo "Use GPU: $RETRIEVER_USE_GPU"
echo "GPU IDs: ${RETRIEVER_GPU_IDS:-N/A}"
echo "GPU Shard: $RETRIEVER_GPU_SHARD"
echo "GPU Load Mode: $RETRIEVER_GPU_LOAD_MODE"
echo ""
echo "Server will be available at: http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
echo "=========================================="

PYTHON_EXEC=${RETRIEVER_PYTHON_EXEC:-python}
export PYTHONUNBUFFERED=1

if ! command -v "$PYTHON_EXEC" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $PYTHON_EXEC"
    exit 1
fi

echo "Using Python: $PYTHON_EXEC"
"$PYTHON_EXEC" -c "import faiss; print(f'Check Faiss: {faiss.__version__}')"

CMD=(
  "$PYTHON_EXEC" "$SCRIPT_DIR/retrieval_server.py"
  --index_path "$INDEX_FILE"
  --corpus_path "$CORPUS_FILE"
  --topk "$RETRIEVER_TOPK"
  --retriever_name "$RETRIEVER_NAME"
  --retriever_model "$RETRIEVER_PATH"
  --port "$RETRIEVER_PORT"
)

if [[ "$USE_GPU" == "1" ]]; then
    CMD+=(--faiss_gpu --gpu_ids "$RETRIEVER_GPU_IDS" --gpu_load_mode "$RETRIEVER_GPU_LOAD_MODE" "$GPU_SHARD_FLAG")
fi

exec "${CMD[@]}"
