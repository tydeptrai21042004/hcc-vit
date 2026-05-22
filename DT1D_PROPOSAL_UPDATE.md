# DT1D / finite weighted h-Hartley--cosine adapter update

This project has been updated so the ViT token adapter matches the current proposal method.

## Implemented proposal behavior

The old HCC token adapter is now a backward-compatible wrapper around `DT1DAdapter` in:

```text
src/models/vit_adapter/hcc_adapter.py
```

The adapter now implements:

- finite weighted h-Hartley--cosine axial depthwise filtering;
- effective axial kernel support length `2*M + 3`;
- optional multi-dilation branches, e.g. `DILATIONS: "1,2,4"`;
- static/global learnable softmax gates over axis--dilation responses;
- group-shared symmetric coefficients through `ALPHA_GROUP`;
- optional grouped pointwise bottleneck mixing through `NO_PW`, `PW_RATIO`, and `PW_GROUPS`;
- scalar residual gate initialized by default at `GATE_INIT: 0.0` for identity-safe insertion.

## Important manuscript wording

The current implementation uses **static/global learnable axis--scale gates**. It does **not** use an input-adaptive/sample-adaptive router. The old flags `INPUT_ADAPTIVE_GATE` and `GATE_REDUCTION` are accepted only for compatibility and are intentionally ignored.

## Main config keys

Example:

```yaml
MODEL:
  TRANSFER_TYPE: "adapter"
  ADAPTER:
    NAME: "HCC"
    HCC:
      M: 1
      H: 1
      AXIS: "hw"
      DILATIONS: "1,2,4"
      SCALE_ADAPTIVE: True
      SEPARATE_AXIS_KERNELS: True
      GATE_TEMPERATURE: 1.0
      INPUT_ADAPTIVE_GATE: False
      ALPHA_GROUP: 16
      PER_CHANNEL: False
      TIE_SYM: True
      NO_PW: True
      USE_PW: False
      PW_RATIO: 32
      PW_GROUPS: 4
      USE_BN: False
      RESIDUAL_SCALE: 1.0
      GATE_INIT: 0.0
      PADDING: "reflect"
```

## Quick method test

```bash
python -m pytest -q tests/test_dt1d_hcc_token_adapter.py
```

Expected result:

```text
4 passed
```
