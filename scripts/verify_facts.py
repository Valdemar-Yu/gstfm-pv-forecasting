#!/usr/bin/env python3
"""Check that the public package contains the files needed for verification."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "data/input_manifest.json",
    ROOT / "paper_results/multi_horizon.csv",
    ROOT / "paper_results/gstfm_seeds.csv",
    ROOT / "paper_results/site1b.csv",
    ROOT / "models/iTransformer_LSTM_agcf.py",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing public verification files: " + ", ".join(missing))

manifest = json.loads((ROOT / "data/input_manifest.json").read_text())
if not manifest:
    raise SystemExit("The input manifest is empty")
for name in ("multi_horizon.csv", "gstfm_seeds.csv", "site1b.csv"):
    with (ROOT / "paper_results" / name).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Aggregate result file is empty: {name}")

raw_csv = list((ROOT / "data").rglob("*.csv"))
if raw_csv:
    raise SystemExit("Raw CSV files must not be included in the public package")
print(f"OK: {len(manifest)} input entries and aggregate result tables are present; raw CSVs are absent.")
