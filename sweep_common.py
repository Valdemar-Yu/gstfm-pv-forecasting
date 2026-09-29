"""Shared plumbing for the GSTFM sweep drivers.

All sweeps follow the same contract: build a default-protocol arg namespace,
overlay the tuned config for that (site, season, horizon), overlay the sweep's
own overrides, run one training, and keep going if a single run dies.
Resumability comes from train_gstfm itself (it skips combos already present in
the results CSV), so a killed sweep can simply be relaunched.
"""

import datetime
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_gstfm import build_arg_parser, apply_best_params, main as train_main  # noqa: E402

SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
SEEDS = [42, 2023, 2024, 2025]
FAILED_LOG = os.path.join(REPO_ROOT, 'results', 'failed_runs.log')


def log_failure(tag, exc):
    os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)
    with open(FAILED_LOG, 'a', encoding='utf-8') as f:
        f.write(f'\n[{datetime.datetime.now().isoformat()}] {tag}\n')
        f.write(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def resolve_out(out):
    return out if os.path.isabs(out) else os.path.join(REPO_ROOT, out)


def build_run_args(site, season, pred_len, seed, device, results_csv,
                   use_tuned=True, **overrides):
    """Default protocol -> tuned config for this cell -> sweep overrides."""
    args = build_arg_parser().parse_args([])
    args.site = site
    args.season = season
    args.pred_len = pred_len
    args.seed = seed
    args.device = device
    args.results_csv = results_csv
    if use_tuned:
        apply_best_params(args)  # no-op with a warning when the JSON is absent
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def run_sweep(cells, device, results_csv, use_tuned=True, label='sweep'):
    """cells: iterable of (site, season, pred_len, seed, overrides_dict)."""
    cells = list(cells)
    total = len(cells)
    t0 = time.time()
    for i, (site, season, pred_len, seed, overrides) in enumerate(cells, 1):
        tag = (f'{label} {site}/{season} h={pred_len} seed={seed} '
               f'{ {k: v for k, v in overrides.items()} }')
        print(f'\n[{i}/{total}] {tag} (elapsed {time.time() - t0:.0f}s)', flush=True)
        try:
            args = build_run_args(site, season, pred_len, seed, device, results_csv,
                                  use_tuned=use_tuned, **overrides)
            train_main(args)
        except Exception as exc:
            log_failure(tag, exc)
            print(f'[{i}/{total}] FAILED {tag}: {exc} (logged to {FAILED_LOG})', flush=True)
    print(f'\n{label} finished in {time.time() - t0:.0f}s. Results: {results_csv}', flush=True)
