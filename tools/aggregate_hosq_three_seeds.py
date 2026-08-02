#!/usr/bin/env python3
"""Aggregate publication-style HOSQ runs over seeds 0, 1, and 2."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


REQUIRED_SEEDS = (0, 1, 2)


def _load_summaries(root: Path):
    candidates = sorted(root.rglob("run_summary.json"))
    by_seed = {}
    for path in candidates:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        seed = int(data["seed"])
        if seed in REQUIRED_SEEDS:
            if seed in by_seed:
                raise RuntimeError(
                    f"Multiple run_summary.json files found for seed {seed}: "
                    f"{by_seed[seed][0]} and {path}"
                )
            by_seed[seed] = (path, data)
    missing = [seed for seed in REQUIRED_SEEDS if seed not in by_seed]
    if missing:
        raise RuntimeError(f"Missing completed seed summaries: {missing}")
    return [by_seed[seed] for seed in REQUIRED_SEEDS]


def _number(data, path):
    current = data
    for key in path.split("."):
        if key not in current:
            return None
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _stats(values):
    values = [float(v) for v in values]
    return {
        "values": values,
        "mean": mean(values),
        "std": stdev(values),
        "formatted": f"{mean(values):.6f} ± {stdev(values):.6f}",
    }


def aggregate(root: Path):
    loaded = _load_summaries(root)
    summaries = [data for _, data in loaded]
    metrics = [
        "best_val_acc1_percent",
        "test.top1",
        "test.top5",
        "test.loss",
        "total_train_time_sec",
        "mean_epoch_time_sec",
        "trainable_parameters",
        "total_parameters",
    ]
    aggregate_result = {
        "seeds": list(REQUIRED_SEEDS),
        "source_files": [str(path) for path, _ in loaded],
        "runs": summaries,
        "metrics": {},
    }
    for metric in metrics:
        values = [_number(summary, metric) for summary in summaries]
        if all(value is not None and math.isfinite(value) for value in values):
            aggregate_result["metrics"][metric] = _stats(values)

    # Evaluator top1/top5 are fractions. Add percentage views without modifying
    # the raw values used by the original repository.
    for raw_key, percent_key in (("test.top1", "test_acc1_percent"), ("test.top5", "test_acc5_percent")):
        if raw_key in aggregate_result["metrics"]:
            values = [100.0 * value for value in aggregate_result["metrics"][raw_key]["values"]]
            aggregate_result["metrics"][percent_key] = _stats(values)
    return aggregate_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root containing seed_0/seed_1/seed_2 summaries")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    result = aggregate(root)
    json_output = args.json_output or root / "hosq_three_seed_summary.json"
    csv_output = args.csv_output or root / "hosq_three_seed_summary.csv"
    json_output.parent.mkdir(parents=True, exist_ok=True)

    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "mean_pm_std"])
        for metric, stats in sorted(result["metrics"].items()):
            writer.writerow([metric, stats["mean"], stats["std"], stats["formatted"]])

    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"Wrote {json_output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
