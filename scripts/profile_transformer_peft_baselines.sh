#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${1:-/path/to/pretrained-vit-root}
OUT_ROOT=${2:-outputs/peft_profiles}
NO_PRETRAIN_FLAG=${NO_PRETRAIN_FLAG:-""}  # set to --no-pretrain for architecture-only profiling
BATCH_SIZE=${BATCH_SIZE:-32}
REPEAT=${REPEAT:-30}

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
  out_dir="${OUT_ROOT}/${cfg_name}"
  echo "==> Profiling ${cfg_name}"
  python tools/profile_efficiency.py \
    --config-file "configs/peft_transformer/${cfg_name}.yaml" \
    --output-dir "${out_dir}" \
    ${NO_PRETRAIN_FLAG} \
    --batch-size "${BATCH_SIZE}" \
    --repeat "${REPEAT}" \
    MODEL.MODEL_ROOT "${MODEL_ROOT}"
done

python tools/aggregate_profiles.py --root "${OUT_ROOT}" --out "${OUT_ROOT}/efficiency_summary.csv"
