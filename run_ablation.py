"""Component ablation for GSTFM -> paper Table 3.

Six variants x 4 seasons x horizons {1, 16} by default.

The two horizons matter. TFCR is a Haar time-frequency consistency term over
the predicted sequence; with a single-step output (H=1) the one-level Haar
decomposition has nothing to decompose and WaveletConsistencyLoss returns the
plain base loss (utils/losses.py short-circuits on y.shape[1] < 2). So at H=1
the 'w/o TFCR' row is *expected* to equal the full model — that is a property
of the operator, not a null result — while at H=16 the term is live and its
contribution is measurable. Running both makes that explicit in the paper
instead of leaving a claim the code cannot support.

Variants
    full       proposed model: AGCF gate + KAN head + TFCR
    wo_agcf    gate removed (gate_type='none')
    wo_tfcr    lambda_A = lambda_D = 0  (loss degenerates to plain MAE)
    wo_both    gate removed and TFCR off
    mlp_head   KAN readout replaced by a 2-layer MLP
    se_gate    AGCF replaced by a standard squeeze-excitation gate
"""

import argparse
import os

from sweep_common import SEASONS, resolve_out, run_sweep

VARIANTS = {
    'full':     dict(gate_type='agcf',  head='kan'),
    'wo_agcf':  dict(gate_type='none', head='kan'),
    'wo_tfcr':  dict(gate_type='agcf',  head='kan', lambda_A=0.0, lambda_D=0.0),
    'wo_both':  dict(gate_type='none', head='kan', lambda_A=0.0, lambda_D=0.0),
    'mlp_head': dict(gate_type='agcf',  head='mlp'),
    'se_gate':  dict(gate_type='se',   head='kan'),
}


def main():
    p = argparse.ArgumentParser(description='GSTFM component ablation (paper Table 3).')
    p.add_argument('--site', type=str, default='7-First-Solar', choices=['7-First-Solar', '1B'])
    p.add_argument('--seasons', nargs='+', default=SEASONS, choices=SEASONS)
    p.add_argument('--pred_lens', nargs='+', type=int, default=[1, 16])
    p.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default=os.path.join('results', 'ablation.csv'))
    args = p.parse_args()

    out = resolve_out(args.out)
    cells = [
        (args.site, season, h, args.seed,
         dict(variant=name, **VARIANTS[name]))
        for h in args.pred_lens
        for season in args.seasons
        for name in args.variants
    ]
    run_sweep(cells, args.device, out, label='ablation')


if __name__ == '__main__':
    main()
