#!/usr/bin/env python3
"""Aggregate efficiency_profile.json and convergence_summary.json files into one CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="revision_efficiency_summary.csv")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for prof in root.rglob("efficiency_profile.json"):
        row = load_json(prof)
        row["profile_path"] = str(prof)
        conv = load_json(prof.parent / "convergence_summary.json")
        row.update({f"train_{k}": v for k, v in conv.items()})
        rows.append(row)
    if not rows:
        raise SystemExit(f"No efficiency_profile.json files found under {root}")
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {args.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
