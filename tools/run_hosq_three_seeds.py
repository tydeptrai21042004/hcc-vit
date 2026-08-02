#!/usr/bin/env python3
"""Run HOSQ-DT1D with seeds 0, 1, and 2, then aggregate results."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REQUIRED_SEEDS = (0, 1, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config overrides after '--', e.g. DATA.DATAPATH /kaggle/input/...",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_file = Path(args.config_file)
    if not config_file.is_absolute():
        config_file = (repo_root / config_file).resolve()
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extra = list(args.opts)
    if extra and extra[0] == "--":
        extra = extra[1:]

    for seed in REQUIRED_SEEDS:
        existing = list(output_root.rglob(f"seed_{seed}/run_summary.json"))
        if args.skip_complete and len(existing) == 1:
            print(f"[seed {seed}] complete: {existing[0]}")
            continue
        if len(existing) > 1:
            raise RuntimeError(f"Multiple existing summaries for seed {seed}: {existing}")

        command = [
            args.python,
            str(repo_root / "train.py"),
            "--config-file",
            str(config_file),
            "OUTPUT_DIR",
            str(output_root),
            "SEED",
            str(seed),
            "MODEL.ADAPTER.NAME",
            "HOSQ",
        ] + extra
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=repo_root, check=True)

    aggregate_command = [
        args.python,
        str(repo_root / "tools" / "aggregate_hosq_three_seeds.py"),
        str(output_root),
    ]
    subprocess.run(aggregate_command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
