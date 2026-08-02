"""Check that HOSQ aggregation requires exactly seeds 0, 1, and 2."""
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "aggregate_hosq_direct",
    _ROOT / "tools" / "aggregate_hosq_three_seeds.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def test_three_seed_aggregation_uses_sample_standard_deviation(tmp_path):
    values = [90.0, 91.0, 92.0]
    for seed, value in enumerate(values):
        folder = tmp_path / f"seed_{seed}"
        folder.mkdir()
        data = {
            "seed": seed,
            "best_val_acc1_percent": value,
            "total_train_time_sec": 10.0 + seed,
            "mean_epoch_time_sec": 1.0,
            "trainable_parameters": 100,
            "total_parameters": 1000,
            "test": {"top1": value / 100.0, "top5": 0.99, "loss": 0.4},
        }
        (folder / "run_summary.json").write_text(json.dumps(data))

    result = _MOD.aggregate(tmp_path)
    assert result["seeds"] == [0, 1, 2]
    assert result["metrics"]["test_acc1_percent"]["mean"] == 91.0
    assert result["metrics"]["test_acc1_percent"]["std"] == 1.0
