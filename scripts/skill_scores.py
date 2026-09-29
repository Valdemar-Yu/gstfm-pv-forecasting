#!/usr/bin/env python3
"""Compute MAE/RMSE skill scores relative to naive persistence.

Input is a dual-scale result CSV with one row per (site, season, model,
forecast horizon, seed, scale). The score is 1 - model_error/persistence_error;
positive values indicate improvement over the reference.
"""
import argparse, csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv_path'); p.add_argument('--out', default='skill_scores.csv'); a=p.parse_args()
    rows=list(csv.DictReader(open(a.csv_path, newline='')))
    key=lambda r:(r['site'],r['season'],r['pred_len'],r['seed'],r['scale'])
    ref={key(r):r for r in rows if r['model'].lower()=='persistence'}
    out=[]
    for r in rows:
        if r['model'].lower()=='persistence': continue
        b=ref.get(key(r))
        if not b: continue
        out.append({**{k:r[k] for k in ('site','season','model','pred_len','seed','scale')},
                    'mae_skill':1-float(r['mae'])/float(b['mae']),
                    'rmse_skill':1-float(r['rmse'])/float(b['rmse'])})
    with open(a.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'wrote {len(out)} rows to {a.out}')
if __name__=='__main__': main()
