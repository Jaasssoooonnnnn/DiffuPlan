#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'MODEL_PATH=/path/to/base SFT_DATA=/path/to/mixed_sft.jsonl NPROC_PER_NODE=8 bash scripts/train_sft.sh [trainer overrides]'
    exit 0
fi
PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-$ROOT/weights/base}
SFT_DATA=${SFT_DATA:-$ROOT/data/sft/mixed_sft.jsonl}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/sft}
[[ -d "$MODEL_PATH" && -f "$SFT_DATA" ]] || { echo 'Set MODEL_PATH and SFT_DATA to existing inputs.' >&2; exit 1; }
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
    echo 'OUT_DIR is not empty; choose a new output directory.' >&2
    exit 1
fi
export PYTHONPATH="$ROOT/SFT/dFactory/VeOmni:$ROOT/SFT/dFactory:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT/SFT/dFactory"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
    tasks/train_llada2_bd.py "$ROOT/configs/sft.yaml" \
    --model.config_path "$ROOT/SFT/dFactory/configs/model_configs/llada2_mini" \
    --model.model_path "$MODEL_PATH" --model.tokenizer_path "$MODEL_PATH" \
    --data.train_path "$SFT_DATA" --train.output_dir "$OUT_DIR" "$@"
