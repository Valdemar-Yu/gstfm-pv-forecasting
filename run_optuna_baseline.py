"""Validation-selected Optuna search for six trainable baselines.

The archived protocol uses 30 trials per model and season at H=1.
Persistence has no trainable hyperparameters."""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import optuna  # noqa: E402

from train_baseline import (  # noqa: E402
    build_arg_parser, tuned_arch_path, main as train_main,
)

SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
TUNABLE = ['dlinear', 'timesnet', 'fedformer', 'lstm', 'itransformer', 'itransformer_lstm']


def suggest(model, trial):
    """Per-model search space, sized for hourly seasonal PV data (~2k train rows,
    seq_len=24, enc_in=5). Ranges are deliberately wide enough that a model
    cannot be dismissed as undertuned."""
    if model == 'dlinear':
        return dict(moving_avg=trial.suggest_categorical('moving_avg', [3, 5, 13, 25]))
    if model == 'timesnet':
        return dict(
            d_model=trial.suggest_categorical('d_model', [16, 32, 64, 128]),
            d_ff=trial.suggest_categorical('d_ff', [32, 64, 128, 256]),
            e_layers=trial.suggest_categorical('e_layers', [1, 2, 3]),
            top_k=trial.suggest_categorical('top_k', [2, 3, 5]),
            num_kernels=trial.suggest_categorical('num_kernels', [3, 4, 6]),
        )
    if model == 'fedformer':
        return dict(
            d_model=trial.suggest_categorical('d_model', [16, 32, 64, 128, 256]),
            d_ff=trial.suggest_categorical('d_ff', [32, 64, 128, 256, 512]),
            e_layers=trial.suggest_categorical('e_layers', [1, 2, 3]),
            d_layers=trial.suggest_categorical('d_layers', [1, 2]),
            n_heads=trial.suggest_categorical('n_heads', [2, 4, 8]),
            moving_avg=trial.suggest_categorical('moving_avg', [3, 5, 13, 25]),
            modes=trial.suggest_categorical('modes', [8, 16, 32, 64]),
            mode_select=trial.suggest_categorical('mode_select', ['random', 'low']),
        )
    if model == 'lstm':
        return dict(
            hidden_size=trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256]),
            num_layers=trial.suggest_categorical('num_layers', [1, 2, 3, 4]),
        )
    if model == 'itransformer':
        return dict(
            dim=trial.suggest_categorical('dim', [16, 32, 64, 128, 256]),
            depth=trial.suggest_categorical('depth', [1, 2, 3, 4, 5, 6]),
            heads=trial.suggest_categorical('heads', [2, 4, 6, 8, 12, 16]),
        )
    if model == 'itransformer_lstm':
        # same space as the proposed model's backbone, so the ablation of the
        # gate/head is not confounded by an unequal search
        return dict(
            dim_embed=trial.suggest_categorical('dim_embed', [16, 32, 64, 128, 256]),
            depth=trial.suggest_categorical('depth', [1, 2, 3, 4, 5, 6]),
            heads=trial.suggest_categorical('heads', [2, 4, 6, 8, 12, 16]),
            dim_lstm=trial.suggest_categorical('dim_lstm', [16, 32, 64, 128, 256]),
            depth_lstm=trial.suggest_categorical('depth_lstm', [1, 2, 3, 4]),
        )
    raise ValueError(f'no search space for {model}')


def make_objective(model, site, season, pred_len, seed, device, trials_csv):
    def objective(trial):
        params = suggest(model, trial)
        run_args = build_arg_parser().parse_args([])  # protocol defaults
        run_args.model = model
        run_args.site = site
        run_args.season = season
        run_args.pred_len = pred_len
        run_args.seed = seed
        run_args.device = device
        run_args.out = trials_csv
        run_args.use_tuned = False       # the trial IS the config
        run_args.arch = params
        # Keep trial checkpoints out of results/ckpt/. Every trial shares the
        # same (model, site, season, h, seed) key and would otherwise overwrite
        # the production checkpoint, leaving the LAST trial's model sitting
        # under the name downstream scripts treat as the tuned one.
        run_args.save_dir = os.path.join('results', 'ckpt', 'optuna')
        out = train_main(run_args)
        trial.set_user_attr('test_mae_std', float(out['std']['mae']))
        return float(out['valid']['mae'])

    return objective


def main():
    p = argparse.ArgumentParser(description='Optuna search for the baselines (validation-selected).')
    p.add_argument('--models', nargs='+', default=TUNABLE, choices=TUNABLE)
    p.add_argument('--site', type=str, default='7-First-Solar', choices=['7-First-Solar', '1B'])
    p.add_argument('--seasons', nargs='+', default=SEASONS, choices=SEASONS)
    p.add_argument('--pred_len', type=int, default=1)
    p.add_argument('--n_trials', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--storage', type=str,
                   default=os.path.join(REPO_ROOT, 'results', 'optuna_baselines.db'))
    args = p.parse_args()

    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
    storage = f'sqlite:///{args.storage}'
    trials_csv = os.path.join(REPO_ROOT, 'results', 'optuna_baseline_trials.csv')
    t0 = time.time()

    for model in args.models:
        for season in args.seasons:
            study_name = f'{model}_{args.site}_{season}_h{args.pred_len}'
            study = optuna.create_study(
                direction='minimize', load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=storage, study_name=study_name)
            done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
            todo = max(0, args.n_trials - done)
            print(f'\n=== {study_name}: {done} done, running {todo} more '
                  f'(elapsed {time.time() - t0:.0f}s) ===', flush=True)
            if todo:
                study.optimize(
                    make_objective(model, args.site, season, args.pred_len,
                                   args.seed, args.device, trials_csv),
                    n_trials=todo)

            payload = dict(study.best_params)
            payload['_study'] = study_name
            payload['_valid_mae'] = study.best_value
            payload['_n_trials'] = len(study.trials)
            out_path = tuned_arch_path(model, args.site, season, args.pred_len)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            print(f'[best] {study_name} valid_mae={study.best_value:.4f} -> {out_path}')
            print(f'       {study.best_params}', flush=True)

    print(f'\nSearch finished in {time.time() - t0:.0f}s. '
          f'Per-trial metrics: {trials_csv}', flush=True)


if __name__ == '__main__':
    main()
