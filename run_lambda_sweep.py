"""TFCR weight sensitivity for GSTFM -> paper Fig. 3.

Five (lambda_A, lambda_D) combinations x 4 seasons, at H=16 by default.

H=1 is deliberately NOT the default. WaveletConsistencyLoss returns the plain
base loss when the prediction has a single step (utils/losses.py), so at H=1
every lambda produces a bit-identical loss and the sweep is vacuous: the five
"sensitivity" points would differ only by nondeterminism. Pass --pred_lens 1
if you want to demonstrate exactly that (the five rows should come out equal).
"""

import argparse
import os

from sweep_common import SEASONS, resolve_out, run_sweep

LAMBDAS = [(0.05, 0.05), (0.10, 0.10), (0.20, 0.20), (0.10, 0.20), (0.20, 0.10)]


def main():
    p = argparse.ArgumentParser(description='TFCR lambda sensitivity sweep (paper Fig. 3).')
    p.add_argument('--site', type=str, default='7-First-Solar', choices=['7-First-Solar', '1B'])
    p.add_argument('--seasons', nargs='+', default=SEASONS, choices=SEASONS)
    p.add_argument('--pred_lens', nargs='+', type=int, default=[16])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default=os.path.join('results', 'lambda_sweep.csv'))
    args = p.parse_args()

    out = resolve_out(args.out)
    cells = [
        (args.site, season, h, args.seed,
         dict(variant=f'lam{la:.2f}-{ld:.2f}', lambda_A=la, lambda_D=ld,
              gate_type='agcf', head='kan'))
        for h in args.pred_lens
        for season in args.seasons
        for (la, ld) in LAMBDAS
    ]
    run_sweep(cells, args.device, out, label='lambda')


if __name__ == '__main__':
    main()
