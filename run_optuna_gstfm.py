"""Validation-selected Optuna search for GSTFM.

The archived protocol uses 40 trials per season at H=1. TFCR coefficients
are fixed because the wavelet terms vanish for a single-step output."""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import optuna  # noqa: E402

from train_gstfm import build_arg_parser, best_params_path, main as train_main  # noqa: E402

SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']


def suggest(trial):
    """Round-1 backbone space, plus the gate's internal knobs.

    `use_agcf` is deliberately NOT searched, unlike round 1 (Quasa.py:133). The
    gate is the paper's contribution, not a hyper-parameter: if a season's best
    trial happened to land on use_agcf=False, the model called "GSTFM" in the
    seed table would be ungated while the one called "GSTFM" in the ablation
    table is gated, and the 'w/o AGCF' ablation row would equal the full row by
    construction. The gate is always on here; whether it helps is what
    run_ablation.py measures.
    """
    return dict(
        dim_embed=trial.suggest_categorical('dim_embed', [16, 32, 64, 128, 256]),
        depth_lstm=trial.suggest_categorical('depth_lstm', [1, 2, 3, 4]),
        depth=trial.suggest_categorical('depth', [1, 2, 3, 4, 5, 6]),
        heads=trial.suggest_categorical('heads', [2, 4, 6, 8, 12, 16]),
        dim_lstm=trial.suggest_categorical('dim_lstm', [16, 32, 64, 128, 256]),
        gate_width=trial.suggest_categorical('gate_width', [4, 6, 8]),
        gate_depth=trial.suggest_categorical('gate_depth', [2, 3, 4]),
        gate_channel=bool(trial.suggest_categorical('gate_channel', [0, 1])),
    )


def make_objective(args, trials_csv):
    def objective(trial):
        params = suggest(trial)
        run_args = build_arg_parser().parse_args([])  # protocol defaults
        run_args.site = args.site
        run_args.season = args.season
        run_args.pred_len = args.pred_len
        run_args.seed = args.seed
        run_args.device = args.device
        run_args.lambda_A = args.lambda_A
        run_args.lambda_D = args.lambda_D
        run_args.variant = f'optuna-t{trial.number}'
        run_args.results_csv = trials_csv
        run_args.force = True
        run_args.gate_type = 'agcf'   # the gate is fixed on; see suggest()
        for k, v in params.items():
            setattr(run_args, k, v)

        out = train_main(run_args)
        if out is None:
            raise optuna.TrialPruned('trainer returned no metrics')
        trial.set_user_attr('test_mae_std', float(out['test_mae_std']))
        return float(out['valid_mae'])

    return objective


def main():
    p = argparse.ArgumentParser(description='Optuna search for GSTFM (validation-selected).')
    p.add_argument('--site', type=str, default='7-First-Solar', choices=['7-First-Solar', '1B'])
    p.add_argument('--seasons', nargs='+', default=SEASONS, choices=SEASONS)
    p.add_argument('--pred_len', type=int, default=1)
    p.add_argument('--n_trials', type=int, default=40)
    p.add_argument('--seed', type=int, default=42, help='training seed used inside each trial')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--lambda_A', type=float, default=0.05)
    p.add_argument('--lambda_D', type=float, default=0.05)
    p.add_argument('--storage', type=str,
                   default=os.path.join(REPO_ROOT, 'results', 'optuna_gstfm.db'))
    args = p.parse_args()

    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
    storage = f'sqlite:///{args.storage}'
    trials_csv = os.path.join(REPO_ROOT, 'results', 'optuna_trials.csv')
    t0 = time.time()

    for season in args.seasons:
        sub = argparse.Namespace(**vars(args))
        sub.season = season
        study_name = f'gstfm_{args.site}_{season}_h{args.pred_len}'
        study = optuna.create_study(
            direction='minimize', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            storage=storage, study_name=study_name)

        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        todo = max(0, args.n_trials - done)
        print(f'\n=== {study_name}: {done} completed, running {todo} more '
              f'(elapsed {time.time() - t0:.0f}s) ===', flush=True)
        if todo:
            study.optimize(make_objective(sub, trials_csv), n_trials=todo)

        out_path = best_params_path(args.site, season, args.pred_len)
        payload = dict(study.best_params)
        payload['gate_channel'] = bool(payload.get('gate_channel', 0))
        payload['_study'] = study_name
        payload['_valid_mae'] = study.best_value
        payload['_n_trials'] = len(study.trials)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f'[best] {study_name} valid_mae={study.best_value:.4f} -> {out_path}')
        print(f'       {study.best_params}', flush=True)

    print(f'\nSearch finished in {time.time() - t0:.0f}s. '
          f'Per-trial metrics: {trials_csv}', flush=True)


if __name__ == '__main__':
    main()
