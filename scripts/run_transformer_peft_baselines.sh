#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${1:-/path/to/flowers-102}
MODEL_ROOT=${2:-/path/to/pretrained-vit-root}
OUT_ROOT=${3:-outputs/peft_transformer}
SEEDS=${SEEDS:-"42 44 82"}
EPOCHS=${EPOCHS:-50}

CONFIGS=(
  flowers_linear
  flowers_bitfit
  flowers_vpt
  flowers_pfeiffer
  flowers_lora
  flowers_adaptformer
  flowers_ssf
  flowers_hcc_dt1d
  flowers_full
)

for cfg_name in "${CONFIGS[@]}"; do
  for seed in ${SEEDS}; do
    out_dir="${OUT_ROOT}/${cfg_name}/seed_${seed}"
    echo "==> Running ${cfg_name}, seed=${seed}"
    python train.py \
      --config-file "configs/peft_transformer/${cfg_name}.yaml" \
      DATA.DATAPATH "${DATA_PATH}" \
      MODEL.MODEL_ROOT "${MODEL_ROOT}" \
      OUTPUT_DIR "${out_dir}" \
      SEED "${seed}" \
      SOLVER.TOTAL_EPOCH "${EPOCHS}"
  done
done
