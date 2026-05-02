# Transformer PEFT revision code

This branch is transformer-only. It adds reviewer-facing baselines and profiling tools without adding CNN-specific baselines.

## Baselines included

| Method | Config | Notes |
|---|---|---|
| Linear probe | `configs/peft_transformer/flowers_linear.yaml` | frozen ViT backbone + classifier head |
| Full fine-tuning | `configs/peft_transformer/flowers_full.yaml` | reviewer-requested upper bound |
| BitFit | `configs/peft_transformer/flowers_bitfit.yaml` | bias-only PEFT |
| VPT | `configs/peft_transformer/flowers_vpt.yaml` | visual prompt tuning |
| Pfeiffer adapter | `configs/peft_transformer/flowers_pfeiffer.yaml` | classic bottleneck adapter |
| LoRA | `configs/peft_transformer/flowers_lora.yaml` | attention query/value low-rank adaptation |
| AdaptFormer | `configs/peft_transformer/flowers_adaptformer.yaml` | ViT adapter baseline |
| SSF | `configs/peft_transformer/flowers_ssf.yaml` | scale-and-shift feature PEFT baseline |
| HCC / DT1D | `configs/peft_transformer/flowers_hcc_dt1d.yaml` | corrected adapter based on the 1D-DT implementation |

## Run training

```bash
pip install -r requirements.txt

SEEDS="42 44 82" EPOCHS=50 bash scripts/run_transformer_peft_baselines.sh \
  /path/to/flowers-102 \
  /path/to/pretrained-vit-root \
  outputs/peft_transformer
```

The model root should contain the ViT checkpoint used by the original repo, for example:

```text
ViT-B_16-224.npz
```

## Profile FLOPs, memory, latency, and parameter counts

For architecture-only profiling without loading pretrained weights:

```bash
NO_PRETRAIN_FLAG="--no-pretrain" BATCH_SIZE=32 REPEAT=30 \
  bash scripts/profile_transformer_peft_baselines.sh \
  /path/to/pretrained-vit-root \
  outputs/peft_profiles
```

For final paper profiling with pretrained weights:

```bash
BATCH_SIZE=32 REPEAT=30 bash scripts/profile_transformer_peft_baselines.sh \
  /path/to/pretrained-vit-root \
  outputs/peft_profiles
```

Each method writes:

```text
efficiency_profile.json
efficiency_profile.csv
```

The aggregate table is saved as:

```text
outputs/peft_profiles/efficiency_summary.csv
```

Metrics include:

- total parameters
- trainable parameters
- trainable percentage
- FLOPs / GFLOPs when `fvcore` is installed
- inference latency
- inference throughput
- peak inference memory
- dummy training-step time
- peak training memory

## Convergence and training-time outputs

During training, the trainer now saves:

```text
history.json
convergence_summary.json
```

These include:

- epoch time
- average batch time
- training loss
- validation top-1
- test top-1 when available
- peak training memory
- best validation epoch

## Tests

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
```

The tests verify:

- corrected HCC/DT1D grouping and identity-safe initialization
- H/W response averaging in `axis="hw"`
- class token preservation
- LoRA replacement and base-projection freezing
- AdaptFormer and SSF module behavior
- no ConvAdapter config appears in this transformer-only baseline set
