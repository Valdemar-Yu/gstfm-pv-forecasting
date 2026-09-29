"""Cross-site experiment on site 1B (referee 3, comment 2).

Sweeps all baseline models plus the full GSTFM model at h=1 over the four
seasons of site 1B. Baselines are run in-process through train_baseline.main;
GSTFM is run as a subprocess of train_gstfm.py. Results are appended to
results/site1b.csv (header written only if the file is new) and combos that
are already present in the CSV are skipped, so an interrupted sweep resumes
cleanly.

At the end the script prints the per-season sigma_train of site 1B (std of
the >=0-clipped Active_Power over the first 80% of each season CSV, ddof=0,
i.e. exactly what StandardScaler/scalar_y uses), which defines the
percentage scale for 1B in the paper (percent value = metric_kw / sigma_train * 100).

Run from the repo root:
    python run_site1b.py [--seed 42] [--models ...] [--sigma_only]
"""

import argparse
import os
import subprocess
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
# Models handled by train_baseline.main; 'gstfm' is handled by train_gstfm.py.
BASELINE_MODELS = [
    "persistence",
    "dlinear",
    "timesnet",
    "fedformer",
    "itransformer",
    "lstm",
    "itransformer_lstm",
]
ALL_MODELS = BASELINE_MODELS + ["gstfm"]

# Year suffix of the season CSV files per site.
SITE_YEARS = {"7-First-Solar": "2019_2022", "1B": "2019_2023"}

RESULTS_COLUMNS = [
    "site", "season", "model", "pred_len", "seed", "scale",
    "mae", "rmse", "r2", "mbe", "epochs_run", "train_seconds",
]


def season_csv_path(site: str, season: str) -> str:
    years = SITE_YEARS[site]
    return os.path.join(REPO_ROOT, "data", site, f"{season}_{site}_{years}_h.csv")


def load_done_combos(results_csv: str):
    """Return the set of (site, season, model, pred_len, seed) already in the CSV."""
    done = set()
    if not os.path.isfile(results_csv):
        return done
    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:  # unreadable / empty file -> treat as no history
        print(f"[resume] could not read {results_csv} ({exc}); starting fresh")
        return done
    needed = {"site", "season", "model", "pred_len", "seed"}
    if not needed.issubset(df.columns):
        print(f"[resume] {results_csv} lacks key columns; ignoring for resume")
        return done
    for _, row in df.iterrows():
        # Model names are compared case-insensitively: train_gstfm.py records
        # the model as 'GSTFM' while this driver names it 'gstfm'.
        done.add((str(row["site"]), str(row["season"]), str(row["model"]).lower(),
                  int(row["pred_len"]), int(row["seed"])))
    return done


def call_baseline(model: str, site: str, season: str, pred_len: int, seed: int,
                  results_csv: str, extra_args=None) -> bool:
    """Run one baseline config through train_baseline.main. Returns True on success.

    train_baseline.main(args) takes a parsed argparse.Namespace, so we build it
    with train_baseline.build_arg_parser(). The results CSV flag there is --out.
    """
    import train_baseline  # deferred: needs torch, only on the server

    argv = [
        "--site", site,
        "--season", season,
        "--model", model,
        "--pred_len", str(pred_len),
        "--seed", str(seed),
        "--out", results_csv,
    ]
    if extra_args:
        argv += list(extra_args)
    train_baseline.main(train_baseline.build_arg_parser().parse_args(argv))
    return True


def call_gstfm(site: str, season: str, pred_len: int, seed: int,
               results_csv: str, extra_args=None) -> bool:
    """Run the full GSTFM model via train_gstfm.py as a subprocess."""
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "train_gstfm.py"),
        "--site", site,
        "--season", season,
        "--pred_len", str(pred_len),
        "--seed", str(seed),
        "--results_csv", results_csv,
    ]
    if extra_args:
        cmd += list(extra_args)
    print("[gstfm] " + " ".join(cmd))
    ret = subprocess.run(cmd, cwd=REPO_ROOT)
    return ret.returncode == 0


def sigma_train_per_season(site: str, train_frac: float = 0.8):
    """sigma_train per season: std (ddof=0) of clipped Active_Power on the train split.

    Mirrors data.data_loader.split_data_cnn: target clipped >= 0 before
    scaling, StandardScaler fit on the first int(0.8 * N) rows (population std).
    """
    sigmas = {}
    for season in SEASONS:
        path = season_csv_path(site, season)
        if not os.path.isfile(path):
            print(f"[sigma] missing CSV: {path}")
            continue
        df = pd.read_csv(path, header=0)
        y = np.maximum(df["Active_Power"].to_numpy(dtype=float), 0.0)
        n_train = int(len(y) * train_frac)
        # nanstd: StandardScaler ignores NaNs when fitting (the loader's
        # interpolate call is a no-op, so NaNs can reach the scaler).
        sigmas[season] = float(np.nanstd(y[:n_train], ddof=0))
    return sigmas


def print_sigma_report(site: str):
    sigmas = sigma_train_per_season(site)
    print()
    print(f"=== sigma_train of site {site} (train split = first 80%, target clipped >=0, ddof=0) ===")
    for season, sig in sigmas.items():
        print(f"  {season:>7s}: sigma_train = {sig:.6f} kW")
    if sigmas:
        vals = np.array(list(sigmas.values()))
        print(f"  mean = {vals.mean():.6f} kW, min = {vals.min():.6f} kW, max = {vals.max():.6f} kW")
    print("  Percentage scale for the paper: percent_value = metric_kw / sigma_train * 100")
    print("  (equivalently metric_std * 100, since kw = std * sigma_train).")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross-site sweep on site 1B: all baselines + GSTFM at h=1 x 4 seasons.")
    parser.add_argument("--site", default="1B", choices=list(SITE_YEARS),
                        help="site to run (default 1B)")
    parser.add_argument("--seasons", nargs="+", default=SEASONS, choices=SEASONS,
                        help="seasons to run")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        help="models to run")
    parser.add_argument("--pred_len", type=int, default=1, help="forecast horizon (default 1)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="training device passed to every model, e.g. cuda:1")
    parser.add_argument("--results_csv", default=os.path.join("results", "site1b.csv"),
                        help="output CSV (relative to repo root)")
    parser.add_argument("--sigma_only", action="store_true",
                        help="only print the per-season sigma_train report, no training")
    args = parser.parse_args(argv)

    results_csv = args.results_csv
    if not os.path.isabs(results_csv):
        results_csv = os.path.join(REPO_ROOT, results_csv)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    if args.sigma_only:
        print_sigma_report(args.site)
        return

    failures = []
    for model in args.models:
        for season in args.seasons:
            # Re-read on every combo: train_baseline also appends to this file.
            done = load_done_combos(results_csv)
            key = (args.site, season, model, args.pred_len, args.seed)
            if key in done:
                print(f"[skip] already in {os.path.basename(results_csv)}: "
                      f"{model} / {season} / h={args.pred_len} / seed={args.seed}")
                continue
            print(f"[run ] {model} / site={args.site} / {season} / "
                  f"h={args.pred_len} / seed={args.seed}")
            try:
                if model == "gstfm":
                    ok = call_gstfm(args.site, season, args.pred_len, args.seed,
                                    results_csv, ["--device", args.device])
                else:
                    ok = call_baseline(model, args.site, season, args.pred_len,
                                       args.seed, results_csv, ["--device", args.device])
                if not ok:
                    failures.append(key)
            except Exception:
                traceback.print_exc()
                failures.append(key)

    if failures:
        print("\n[warn] failed combos (rerun this script to retry, finished ones are skipped):")
        for key in failures:
            print(f"  {key}")

    # sigma_train report so the paper can define the percentage scale for 1B
    print_sigma_report(args.site)


if __name__ == "__main__":
    main()
