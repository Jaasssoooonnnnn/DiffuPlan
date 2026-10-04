#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'MODEL_PATH=/path/to/sft NPROC_PER_NODE=4 bash scripts/train_rl.sh [OmegaConf overrides]'
    exit 0
fi
PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-$ROOT/weights/sft}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rl}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
[[ -d "$MODEL_PATH" ]] || { echo 'Set MODEL_PATH to the SFT checkpoint directory.' >&2; exit 1; }
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
    echo 'OUT_DIR is not empty; choose a new output directory.' >&2
    exit 1
fi
export PYTHONPATH="$ROOT/Diffuplan:$ROOT/RL:$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
cd "$ROOT"
exec "$PYTHON" -m accelerate.commands.launch \
    --config_file "$ROOT/configs/fsdp.yaml" --num_processes "${NPROC_PER_NODE:-4}" \
    --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
    "$ROOT/RL/rl_editable_plan_fused.py" "config=$ROOT/configs/rl.yaml" \
    "model.pretrained_model=$MODEL_PATH" "sft_anchor.reference_model=$MODEL_PATH" \
    "experiment.project=$OUT_DIR" "search.url=$RETRIEVER_URL" "$@"
