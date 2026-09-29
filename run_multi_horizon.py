"""Multi-horizon baseline sweep (site 7-First-Solar, seed 42).

Sweeps model x pred_len x season and appends dual-scale metrics to
results/multi_horizon.csv (resumable: existing (site,season,model,pred_len,seed)
combos are skipped, so an interrupted sweep can simply be re-launched).

CALIBRATION NOTE: the h=1 rows are the calibration anchor against paper
Table 2 — compare the 'std' rows (paper values / 100) before trusting the
longer horizons (16/32/64).

Per-run failures are logged with a traceback to results/failed_runs.log and
the sweep continues.
"""

import argparse
import datetime
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_baseline import build_arg_parser, load_done_keys, main as train_main  # noqa: E402

DEFAULT_MODELS = ["timesnet", "fedformer", "dlinear", "itransformer_lstm"]
DEFAULT_HORIZONS = [1, 16, 32, 64]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]


def log_failure(log_path, tag, exc):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.datetime.now().isoformat()}] {tag}\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep baselines over horizons {1,16,32,64} x 4 seasons on 7-First-Solar. "
            "h=1 rows serve as CALIBRATION against paper Table 2 (std scale = paper/100)."
        )
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"models to sweep (default: {DEFAULT_MODELS})")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--seasons", nargs="+", default=SEASONS, choices=SEASONS)
    parser.add_argument("--site", type=str, default="7-First-Solar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=os.path.join("results", "multi_horizon.csv"))
    args = parser.parse_args()

    out_path = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    failed_log = os.path.join(REPO_ROOT, "results", "failed_runs.log")

    print("REMINDER: h=1 rows are the calibration anchor against paper Table 2 "
          "(compare 'std'-scale rows to paper values / 100).", flush=True)

    combos = [
        (model, season, h)
        for model in args.models
        for h in args.horizons
        for season in args.seasons
    ]
    total = len(combos)
    done = load_done_keys(out_path)
    t_start = time.time()

    for i, (model, season, h) in enumerate(combos, 1):
        key = (args.site, season, model, h, args.seed)
        elapsed = time.time() - t_start
        if key in done:
            print(f"[{i}/{total}] SKIP (done) model={model} season={season} h={h} "
                  f"elapsed={elapsed:.0f}s", flush=True)
            continue
        print(f"[{i}/{total}] RUN model={model} season={season} h={h} "
              f"seed={args.seed} elapsed={elapsed:.0f}s", flush=True)
        run_args = build_arg_parser().parse_args([])  # protocol defaults
        run_args.model = model
        run_args.site = args.site
        run_args.season = season
        run_args.pred_len = h
        run_args.seed = args.seed
        run_args.device = args.device
        run_args.out = out_path
        try:
            train_main(run_args)
            done.add(key)
        except Exception as exc:  # log and keep sweeping
            tag = f"multi_horizon model={model} site={args.site} season={season} h={h} seed={args.seed}"
            log_failure(failed_log, tag, exc)
            print(f"[{i}/{total}] FAILED {tag}: {exc} (logged to {failed_log})", flush=True)

    print(f"Sweep finished in {time.time() - t_start:.0f}s. Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
