from pathlib import Path
import yaml


def test_peft_configs_do_not_include_convadapter():
    cfg_dir = Path("configs/peft_transformer")
    assert cfg_dir.exists()
    for path in cfg_dir.glob("*.yaml"):
        text = path.read_text().lower()
        assert "convadapter" not in text
        assert "conv_adapter" not in text


def test_new_baseline_configs_exist():
    cfg_dir = Path("configs/peft_transformer")
    for name in ["flowers_lora.yaml", "flowers_adaptformer.yaml", "flowers_ssf.yaml", "flowers_hcc_dt1d.yaml"]:
        assert (cfg_dir / name).exists()


def test_profile_efficiency_tool_contains_reviewer_metrics():
    text = Path("tools/profile_efficiency.py").read_text()
    for key in ["trainable_params", "gflops", "inference_latency_ms", "peak_train_memory_mb"]:
        assert key in text
