#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daylight_metrics.py
===================
Reviewer 3, item 4 (first half): nighttime zero-generation periods inflate the
apparent accuracy of PV forecasting metrics. This script recomputes MAE, RMSE,
R2 and MBE on prediction dumps under three sample filters:

  (a) all       : all hours (matches the numbers currently in the paper);
  (b) daylight  : y_true > zero_threshold (actual-generation hours only);
  (c) clock     : timestamp hour in [day_start, day_end) - a robustness
                  alternative that KEEPS cloudy-but-daylight near-zero samples.

Input : one or more predictions CSVs produced by dump_predictions.py with
        columns  timestamp,y_true_kw,y_pred_kw  (extra columns are ignored;
        common fallback column names are also recognised, see COLUMN_ALIASES).
Output: - formatted console table (per file, per filter) with sample counts,
          % samples removed, metrics, and relative change vs. all-hours;
        - results CSV + ready-to-paste booktabs LaTeX table in ./output/
          (an "output" folder next to this script, override with --out-dir).

Metric definitions (implemented with numpy, matching sklearn and the repo's
utils/tools.py::metrics_of_pv):
    MAE  = mean(|pred - true|)
    RMSE = sqrt(mean((pred - true)^2))
    R2   = 1 - SS_res / SS_tot          (sklearn.metrics.r2_score)
    MBE  = mean(pred - true)            (positive = over-forecast)

Run from anywhere (only needs numpy + pandas, no torch):
    python revision_scripts/daylight_metrics.py \
        --csv "revision_scripts/output/predictions_*.csv" \
        --zero-threshold 0.0 --day-start 6 --day-end 19

NOTE for the paper text: R2 typically DROPS when nighttime zeros are excluded
because the target variance shrinks (the trivially-predictable zeros are
removed) - this is expected and should be anticipated in the discussion, not
treated as a degradation of the model.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration constants (paths are resolved relative to this script so the
# same file runs unchanged on the authors' Linux server).
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
# Default glob: pick up whatever dump_predictions.py wrote to <repo>/results.
DEFAULT_CSV_GLOB = os.path.join(os.path.dirname(SCRIPT_DIR), "results", "predictions_*.csv")

# Accepted column spellings (first match wins). dump_predictions.py writes
# timestamp,y_true_kw,y_pred_kw; the fallbacks make the script reusable for
# dumps coming from the server-side Time-Series-Library baselines as well.
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "date", "time", "datetime"],
    "y_true": ["y_true_kw", "y_true", "true", "trues", "actual"],
    "y_pred": ["y_pred_kw", "y_pred", "pred", "preds", "prediction"],
}

FILTER_ORDER = ["all", "daylight", "clock"]


# ----------------------------------------------------------------------------
# Metrics (numpy only, definitions identical to sklearn /
# iTansformer_LSTM_CA_KAN/utils/tools.py::metrics_of_pv)
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Return dict with mae, rmse, r2, mbe for 1-D numpy arrays.

    R2 follows sklearn.metrics.r2_score: 1 - SS_res/SS_tot. If the true
    values are constant (SS_tot == 0) R2 is undefined and NaN is returned.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mbe": np.nan}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mbe = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mbe": mbe}


def pct_change(new, base):
    """Signed percent change of `new` vs `base`; NaN when not meaningful."""
    if base is None or new is None:
        return float("nan")
    if math.isnan(base) or math.isnan(new) or abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (new - base) / abs(base)


def fmt(x, nd=4, suffix=""):
    """Format a float, using 'n/a' for NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}{suffix}"


def fmt_signed_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.1f}%"


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def resolve_column(df, logical_name):
    """Find the actual dataframe column for a logical name via COLUMN_ALIASES."""
    lower_map = {c.lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES[logical_name]:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise KeyError(
        f"Could not find a '{logical_name}' column. Tried "
        f"{COLUMN_ALIASES[logical_name]}, available: {list(df.columns)}"
    )


def load_predictions(path):
    """Load one predictions CSV -> (timestamps Series, y_true, y_pred)."""
    df = pd.read_csv(path)
    ts_col = resolve_column(df, "timestamp")
    yt_col = resolve_column(df, "y_true")
    yp_col = resolve_column(df, "y_pred")
    ts = pd.to_datetime(df[ts_col])  # handles both ISO and '2019/9/1 0:00'
    y_true = df[yt_col].to_numpy(dtype=np.float64)
    y_pred = df[yp_col].to_numpy(dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"  [warn] {path}: dropped {n_drop} rows with NaN values")
    return ts[mask].reset_index(drop=True), y_true[mask], y_pred[mask]


def expand_csv_args(csv_args):
    """Expand repeatable --csv arguments; each may be a literal path or a glob."""
    paths = []
    for pattern in csv_args:
        hits = sorted(glob.glob(pattern))
        if hits:
            paths.extend(hits)
        elif os.path.isfile(pattern):
            paths.append(pattern)
        else:
            print(f"[warn] --csv pattern matched nothing: {pattern}")
    # de-duplicate, keep order
    seen, unique = set(), []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            unique.append(ap)
    return unique


# ----------------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------------
def analyse_file(path, zero_threshold, day_start, day_end):
    """Compute metrics under the three filters for a single predictions CSV.

    Returns a list of row dicts (one per filter).
    """
    ts, y_true, y_pred = load_predictions(path)
    n_all = y_true.size

    masks = {
        # (a) every sample, as reported in the current manuscript
        "all": np.ones(n_all, dtype=bool),
        # (b) actual generation only: strictly above the zero threshold.
        "daylight": y_true > zero_threshold,
        # (c) clock-based daytime window [day_start, day_end) - keeps
        #     cloudy-but-daylight zeros that filter (b) would discard.
        "clock": (ts.dt.hour.to_numpy() >= day_start)
                 & (ts.dt.hour.to_numpy() < day_end),
    }

    base = compute_metrics(y_true, y_pred)  # all-hours reference
    rows = []
    for filt in FILTER_ORDER:
        m = masks[filt]
        n = int(m.sum())
        met = base if filt == "all" else compute_metrics(y_true[m], y_pred[m])
        rows.append({
            "file": os.path.basename(path),
            "filter": filt,
            "n": n,
            "pct_removed": 100.0 * (n_all - n) / n_all if n_all else float("nan"),
            "mae": met["mae"],
            "rmse": met["rmse"],
            "r2": met["r2"],
            "mbe": met["mbe"],
            # relative change vs all-hours (percent for MAE/RMSE; for R2 and
            # MBE an absolute delta is also stored because percent change is
            # misleading for unitless / signed metrics near zero)
            "mae_chg_pct": pct_change(met["mae"], base["mae"]) if filt != "all" else 0.0,
            "rmse_chg_pct": pct_change(met["rmse"], base["rmse"]) if filt != "all" else 0.0,
            "r2_chg_pct": pct_change(met["r2"], base["r2"]) if filt != "all" else 0.0,
            "mbe_chg_pct": pct_change(met["mbe"], base["mbe"]) if filt != "all" else 0.0,
            "r2_delta": met["r2"] - base["r2"] if filt != "all" else 0.0,
            "mbe_delta": met["mbe"] - base["mbe"] if filt != "all" else 0.0,
        })
    return rows


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_console_table(rows):
    header = (f"{'File':<38} {'Filter':<9} {'N':>7} {'Removed':>8} "
              f"{'MAE':>9} {'RMSE':>9} {'R2':>8} {'MBE':>9} "
              f"{'dMAE':>8} {'dRMSE':>8} {'dR2':>8} {'dMBE':>9}")
    print(header)
    print("-" * len(header))
    last_file = None
    for r in rows:
        if last_file is not None and r["file"] != last_file:
            print("-" * len(header))
        last_file = r["file"]
        is_base = r["filter"] == "all"
        print(f"{r['file']:<38} {r['filter']:<9} {r['n']:>7d} "
              f"{fmt(r['pct_removed'], 1, '%'):>8} "
              f"{fmt(r['mae']):>9} {fmt(r['rmse']):>9} "
              f"{fmt(r['r2']):>8} {fmt(r['mbe']):>9} "
              f"{('--' if is_base else fmt_signed_pct(r['mae_chg_pct'])):>8} "
              f"{('--' if is_base else fmt_signed_pct(r['rmse_chg_pct'])):>8} "
              f"{('--' if is_base else fmt(r['r2_delta'], 3)):>8} "
              f"{('--' if is_base else fmt(r['mbe_delta'], 3)):>9}")
    print("-" * len(header))
    print("dMAE/dRMSE: percent change vs all-hours. dR2/dMBE: ABSOLUTE delta "
          "vs all-hours\n(percent change is misleading for unitless/signed "
          "metrics near zero; percent versions are in the CSV).")


def build_latex_table(rows, zero_threshold, day_start, day_end):
    """Ready-to-paste booktabs table with the numbers filled in."""
    filter_label = {
        "all": "All hours",
        "daylight": f"Daylight ($P>{zero_threshold:g}$\\,kW)",
        "clock": f"Clock ({day_start:02d}:00--{day_end:02d}:00)",
    }
    lines = [
        "% ---- Daylight-only / zero-power sensitivity analysis (Reviewer 3, item 4) ----",
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Sensitivity of the evaluation metrics to nighttime "
        "zero-generation samples. ``Daylight'' keeps only samples with "
        "measured power above the zero threshold; ``Clock'' keeps a fixed "
        f"{day_start:02d}:00--{day_end:02d}:00 window. The drop in $R^2$ under "
        "filtering is expected: removing trivially-predictable zeros shrinks "
        "the target variance.}",
        "  \\label{tab:daylight_sensitivity}",
        "  \\begin{tabular}{llrrrrr}",
        "    \\toprule",
        "    Series & Filter & $N$ & MAE (kW) & RMSE (kW) & $R^2$ & MBE (kW) \\\\",
        "    \\midrule",
    ]
    last_file = None
    for r in rows:
        # print the (cleaned) file name only on its first row
        name = os.path.splitext(r["file"])[0].replace("_", "\\_") \
            if r["file"] != last_file else ""
        if last_file is not None and r["file"] != last_file:
            lines.append("    \\addlinespace")
        last_file = r["file"]
        lines.append(
            f"    {name} & {filter_label[r['filter']]} & {r['n']} & "
            f"{fmt(r['mae'], 3)} & {fmt(r['rmse'], 3)} & "
            f"{fmt(r['r2'], 3)} & {fmt(r['mbe'], 3)} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Daylight-only / zero-power sensitivity analysis of PV "
                    "forecasting metrics (Reviewer 3, item 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--csv", action="append", default=None, metavar="PATH_OR_GLOB",
        help="Predictions CSV (columns timestamp,y_true_kw,y_pred_kw) from "
             "dump_predictions.py. Repeatable; each value may be a glob, "
             "e.g. --csv 'output/predictions_*.csv'. "
             f"Default: {DEFAULT_CSV_GLOB}")
    parser.add_argument(
        "--zero-threshold", type=float, default=0.0, metavar="KW",
        help="Daylight filter keeps samples with y_true > this value (kW). "
             "0.0 removes exact zeros/negatives only; a small epsilon such "
             "as 0.01*plant_capacity is also worth trying to drop dawn/dusk "
             "near-zeros.")
    parser.add_argument(
        "--day-start", type=int, default=6, metavar="H",
        help="Clock filter: first hour kept (inclusive), 0-23.")
    parser.add_argument(
        "--day-end", type=int, default=19, metavar="H",
        help="Clock filter: end hour (EXCLUSIVE), i.e. default keeps "
             "06:00-18:59.")
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="Directory for the results CSV and the LaTeX table file.")
    args = parser.parse_args()

    if not (0 <= args.day_start < args.day_end <= 24):
        parser.error("Require 0 <= day-start < day-end <= 24.")

    csv_args = args.csv if args.csv else [DEFAULT_CSV_GLOB]
    paths = expand_csv_args(csv_args)
    if not paths:
        print("No input CSVs found. Run dump_predictions.py first, or point "
              "--csv at existing prediction dumps.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} predictions file(s). "
          f"zero_threshold={args.zero_threshold} kW, "
          f"clock window=[{args.day_start:02d}:00, {args.day_end:02d}:00)\n")

    all_rows = []
    for p in paths:
        print(f"Processing {p}")
        all_rows.extend(analyse_file(p, args.zero_threshold,
                                     args.day_start, args.day_end))
    print()

    # -- console table --------------------------------------------------------
    print_console_table(all_rows)

    # -- artifacts -------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "daylight_metrics_results.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False, float_format="%.6f")

    latex = build_latex_table(all_rows, args.zero_threshold,
                              args.day_start, args.day_end)
    tex_path = os.path.join(args.out_dir, "daylight_metrics_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")

    print(f"\nResults CSV : {csv_path}")
    print(f"LaTeX table : {tex_path}\n")
    print("=== Ready-to-paste LaTeX (booktabs) ===")
    print(latex)

    # -- interpretation hint for the response letter / paper text -------------
    print("\n=== Interpretation hint ===")
    print("R2 typically DROPS when nighttime zeros are excluded: the "
          "zero-power samples are trivially predictable, so removing them "
          "shrinks the target variance (SS_tot) faster than the residual sum "
          "of squares. MAE/RMSE usually INCREASE for the same reason. This is "
          "a property of the evaluation protocol, not of any single model - "
          "since ALL models are re-evaluated under identical filters, the "
          "ranking (and the proposed model's relative advantage) is what the "
          "paper text should emphasise. Anticipate the R2 drop explicitly in "
          "the revised manuscript to preempt reviewer confusion.")


if __name__ == "__main__":
    main()
