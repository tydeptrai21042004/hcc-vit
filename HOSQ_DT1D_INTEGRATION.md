# HOSQ-DT1D integration

This repository supports **Hierarchically Orthogonal Spectral Quotient DT1D-Adapter** in supervised ViT, MAE-ViT, and MoCo-ViT adapter paths.

## Final method

Set:

```yaml
MODEL:
  TRANSFER_TYPE: "adapter"
  ADAPTER:
    NAME: "HOSQ"
    HOSQ:
      AXIS: "hw"
      COARSE_GROUP: 32
      SUBGROUP_SIZE: 8
      RANK4: 1
      RANK8: 2
      NUM_PREFIX_TOKENS: 1
      NO_PW: True
      RESIDUAL_SCALE: 1.0
      GATE_INIT: 0.01
      PADDING: "reflect"
      STRICT_PADDING: True
```

For ViT-B/16, each block has 385 HOSQ parameters:

- coarse quotient: `2 × 24 × 5 = 240`;
- offset-4 details: `2 × 24 × 1 = 48`;
- offset-8 details: `2 × 24 × 2 = 96`;
- residual gate: `1`.

Across twelve blocks, this gives 4,620 adapter parameters, excluding the classification head.

## Mathematical implementation

Each coarse Group-32 kernel uses offsets

```text
0, ±1, ±2, ±4, ±8
```

and each Group-8 refinement is generated from zero-mean orthonormal channel contrasts. Fine spatial corrections use

```text
psi_r = delta_-r + delta_r - 2 delta_0,  r in {4, 8}.
```

The height and width kernels are jointly projected so that, for every channel,

```text
||k_h||_1 + ||k_w||_1 <= 1.
```

The final preset executes one depthwise convolution per enabled axis and does not use a pointwise block.

## Three-seed experiment

```bash
python tools/run_hosq_three_seeds.py \
  --config-file configs/finetune/flowers_hosq_adapter.yaml \
  --output-root ./output_hosq_three_seed \
  -- \
  DATA.DATAPATH /path/to/flowers-102 \
  MODEL.MODEL_ROOT /path/to/pretrained_weights
```

The runner requires seeds `0,1,2`. Each epoch uses validation only. The best-validation trainable state is restored, and the test set is evaluated once. Outputs include:

```text
run_summary.json
hosq_three_seed_summary.json
hosq_three_seed_summary.csv
```

## Boundary rule

The final kernel has radius 8. Reflect padding therefore requires both patch-grid dimensions to be greater than 8. ViT-B/16 at 224 pixels has a `14×14` grid and is valid. For patch-32 models with a `7×7` grid, explicitly use:

```yaml
PADDING: "replicate"
```

The code does not silently switch the boundary operator when `STRICT_PADDING: True`.

## Tests

```bash
pytest -q tests/test_dt1d_hcc_token_adapter.py \
  tests/test_hosq_dt1d_token_adapter.py \
  tests/test_three_seed_aggregation.py
```
