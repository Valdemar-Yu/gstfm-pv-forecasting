"""Seed-robustness sweep for a baseline model (default: timesnet).

Sweeps seed in {42, 2023, 2024, 2025} x 4 seasons at h=1 on 7-First-Solar and
appends dual-scale metrics to results/seed_runs.csv (resumable: existing
(site,season,model,pred_len,seed) combos are skipped).

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

from train_baseline import MODEL_NAMES, build_arg_parser, load_done_keys, main as train_main  # noqa: E402

DEFAULT_SEEDS = [42, 2023, 2024, 2025]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]


def log_failure(log_path, tag, exc):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.datetime.now().isoformat()}] {tag}\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def main():
    parser = argparse.ArgumentParser(
        description="Sweep one baseline over seeds {42,2023,2024,2025} x 4 seasons at h=1."
    )
    parser.add_argument("--model", type=str, default="timesnet", choices=MODEL_NAMES,
                        help="baseline to sweep (default: timesnet)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seasons", nargs="+", default=SEASONS, choices=SEASONS)
    parser.add_argument("--site", type=str, default="7-First-Solar")
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=os.path.join("results", "seed_runs.csv"))
    args = parser.parse_args()

    out_path = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    failed_log = os.path.join(REPO_ROOT, "results", "failed_runs.log")

    combos = [(seed, season) for seed in args.seeds for season in args.seasons]
    total = len(combos)
    done = load_done_keys(out_path)
    t_start = time.time()

    for i, (seed, season) in enumerate(combos, 1):
        key = (args.site, season, args.model, args.pred_len, seed)
        elapsed = time.time() - t_start
        if key in done:
            print(f"[{i}/{total}] SKIP (done) model={args.model} season={season} "
                  f"h={args.pred_len} seed={seed} elapsed={elapsed:.0f}s", flush=True)
            continue
        print(f"[{i}/{total}] RUN model={args.model} season={season} h={args.pred_len} "
              f"seed={seed} elapsed={elapsed:.0f}s", flush=True)
        run_args = build_arg_parser().parse_args([])  # protocol defaults
        run_args.model = args.model
        run_args.site = args.site
        run_args.season = season
        run_args.pred_len = args.pred_len
        run_args.seed = seed
        run_args.device = args.device
        run_args.out = out_path
        try:
            train_main(run_args)
            done.add(key)
        except Exception as exc:  # log and keep sweeping
            tag = (f"seed_baselines model={args.model} site={args.site} season={season} "
                   f"h={args.pred_len} seed={seed}")
            log_failure(failed_log, tag, exc)
            print(f"[{i}/{total}] FAILED {tag}: {exc} (logged to {failed_log})", flush=True)

    print(f"Sweep finished in {time.time() - t_start:.0f}s. Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
