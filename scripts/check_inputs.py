#!/usr/bin/env python3
"""Verify the eight processed CSV inputs against data/input_manifest.json."""
import csv, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'data/input_manifest.json').read_text())
failed=[]
for item in manifest:
    p=ROOT/item['file']
    if not p.exists():
        print(f'MISSING {p}'); failed.append(str(p)); continue
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(newline='') as f: rows=list(csv.DictReader(f))
    checks=[digest==item['sha256'], len(rows)==item['rows'], list(rows[0])==item['columns']]
    print(f"{'OK' if all(checks) else 'MISMATCH'} {p.name}: rows={len(rows)} sha256={digest[:12]}")
    if not all(checks): failed.append(str(p))
if failed:
    raise SystemExit(f'{len(failed)} input file(s) failed verification')
print('All manifest checks passed.')
