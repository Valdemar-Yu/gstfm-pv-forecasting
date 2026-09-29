"""Seed-robustness sweep for GSTFM -> paper Table 4, and the paired half of the
Wilcoxon test requested in review R2-6.

Four seeds (42, 2023, 2024, 2025) x 4 seasons, H=1 by default. Round 1 had no
seed loop at all (Quasa.py hardcodes `seeds = 42`), so the mean +/- std of
Table 4 has to be produced here rather than recovered.

The TimesNet counterpart already exists in results/seed_runs.csv (produced by
run_seed_baselines.py) on the same seeds and seasons, so the 16 (season, seed)
cells pair up one-to-one for scripts/stats_test.py.
"""

import argparse
import os

from sweep_common import SEASONS, SEEDS, resolve_out, run_sweep


def main():
    p = argparse.ArgumentParser(description='GSTFM seed-robustness sweep (paper Table 4).')
    p.add_argument('--site', type=str, default='7-First-Solar', choices=['7-First-Solar', '1B'])
    p.add_argument('--seasons', nargs='+', default=SEASONS, choices=SEASONS)
    p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--pred_lens', nargs='+', type=int, default=[1])
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default=os.path.join('results', 'gstfm_seeds.csv'))
    args = p.parse_args()

    out = resolve_out(args.out)
    # gate_type is pinned rather than left to default: the CSV label 'GSTFM'
    # must mean the same architecture here as in run_ablation.py's 'full'.
    cells = [
        (args.site, season, h, seed, dict(gate_type='agcf', head='kan'))
        for h in args.pred_lens
        for season in args.seasons
        for seed in args.seeds
    ]
    run_sweep(cells, args.device, out, label='gstfm-seeds')


if __name__ == '__main__':
    main()
