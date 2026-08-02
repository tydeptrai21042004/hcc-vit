#!/usr/bin/env bash
set -Eeuo pipefail

FILES=(
  HOSQ_DT1D_INTEGRATION.md
  README.md
  RUN_HOSQ_THREE_SEEDS.sh
  requirements-hosq.txt
  configs/finetune/flowers_hosq_adapter.yaml
  configs/finetune/flowers_hosq_pointwise_ablation.yaml
  src/configs/config.py
  src/data/loader.py
  src/engine/evaluator.py
  src/engine/trainer.py
  src/models/vit_adapter/adapter_block.py
  src/models/vit_adapter/hcc_adapter.py
  src/models/vit_adapter/vit.py
  src/models/vit_adapter/vit_mae.py
  src/models/vit_adapter/vit_moco.py
  src/utils/reproducibility.py
  tests/test_hosq_dt1d_token_adapter.py
  tests/test_three_seed_aggregation.py
  tools/aggregate_hosq_three_seeds.py
  tools/run_hosq_three_seeds.py
  tools/validate_hosq_dt1d.py
  train.py
  tune_vtab.py
)

python tools/validate_hosq_dt1d.py
pytest -q tests/test_dt1d_hcc_token_adapter.py \
  tests/test_hosq_dt1d_token_adapter.py \
  tests/test_three_seed_aggregation.py

git add -- "${FILES[@]}"
if git diff --cached --quiet; then
  echo "No HOSQ changes to commit."
  exit 0
fi
git commit -m "Add HOSQ-DT1D and reproducible three-seed training"
echo "Committed HOSQ-DT1D changes. Push with: git push"
