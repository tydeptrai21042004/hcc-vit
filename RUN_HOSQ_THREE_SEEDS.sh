#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/finetune/flowers_hosq_adapter.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_hosq_three_seed}"
DATA_ROOT="${DATA_ROOT:-}"
MODEL_ROOT="${MODEL_ROOT:-}"

EXTRA=()
[[ -n "$DATA_ROOT" ]] && EXTRA+=(DATA.DATAPATH "$DATA_ROOT")
[[ -n "$MODEL_ROOT" ]] && EXTRA+=(MODEL.MODEL_ROOT "$MODEL_ROOT")

python tools/run_hosq_three_seeds.py \
  --config-file "$CONFIG_FILE" \
  --output-root "$OUTPUT_ROOT" \
  --skip-complete \
  -- "${EXTRA[@]}"
