#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day_plots.py
============
Revision experiment (Editor + Reviewer 2/3): publication-quality
predicted-vs-actual time-series plots for representative days
(clear-sky reference, moderately variable, and highly variable / cloudy day).

Input
-----
One or more prediction CSVs, each with columns:
    timestamp, y_true_kw, y_pred_kw
(lenient aliases accepted: date/datetime/time for timestamp,
 y_true/actual/true for y_true_kw, y_pred/pred/prediction for y_pred_kw).

All CSVs must come from the SAME season/site so that timestamps align.
They are inner-joined on timestamp; a warning is printed if rows are dropped.

Usage (run from anywhere; paths may be relative to the CWD):
    python day_plots.py \
        --csv Proposed=revision_scripts/output/predictions_Autumn_agcf.csv \
        --csv iTransformer-LSTM=revision_scripts/output/predictions_Autumn_base.csv \
        --csv TimesNet=path/to/timesnet_autumn_preds.csv \
        --season Autumn

Day selection (automatic, can be overridden with --dates):
    For every complete calendar day (>= --min-points hourly samples) compute
      (i)   daily energy  E  = sum(y_true)          [kWh, hourly data]
      (ii)  variability index VI = sum(|diff(y_true)|) / max(y_true)
            (days with max ~ 0 are excluded from selection)
      (iii) daylight fraction = share of hours with y_true above a small
            threshold (2% of the global peak) -- reported for context.
    Selected days:
      * "Clear day"             : lowest VI among high-energy days
                                  (energy >= 75th percentile)
      * "Moderately variable"   : the day whose VI is the median VI
      * "Highly variable (cloudy)": highest VI day

Output
------
    output/representative_days_{season}.pdf   (vector, for the manuscript)
    output/representative_days_{season}.png   (300 dpi preview)
    output/representative_days_{season}_data.csv     (plotted series)
    output/representative_days_{season}_selection.csv (per-day stats table)
The selected dates and their VI are also printed so the authors can cite
them in the manuscript text.

Notes
-----
* matplotlib only (Agg backend) -> runs headless on the Linux server.
* No torch dependency: this script only post-processes prediction CSVs
  produced by the other revision scripts / the authors' server codebase
  (e.g. thuml Time-Series-Library exports for TimesNet / FEDformer / DLinear).
"""

import argparse
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend (server-safe), must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configurable constants (defaults; most can be overridden via argparse)
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
MIN_POINTS_PER_DAY = 20        # a "complete" day needs at least this many hours
HIGH_ENERGY_QUANTILE = 0.75    # "high-energy" days = top 25% by daily energy
DAYLIGHT_POWER_FRACTION = 0.02 # y_true > 2% of global peak counts as daylight
ZERO_MAX_GUARD_KW = 1e-6       # daily max below this -> VI undefined, excluded

# Column-name aliases (lower-cased) accepted in the input CSVs
TS_ALIASES = ("timestamp", "date", "datetime", "time", "ds")
TRUE_ALIASES = ("y_true_kw", "y_true", "actual", "true", "ground_truth", "target")
PRED_ALIASES = ("y_pred_kw", "y_pred", "pred", "prediction", "forecast", "yhat")

# Distinct dashed styles / colors for the model curves (colorblind-friendly)
MODEL_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7",
                "#E69F00", "#56B4E9", "#F0E442", "#999999"]
MODEL_LINESTYLES = ["--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)),
                    (0, (1, 1)), (0, (3, 2, 1, 2)), "--", "-."]
MODEL_MARKERS = ["s", "^", "v", "D", "P", "X", "*", "o"]


# ----------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ----------------------------------------------------------------------------
def set_pub_style():
    """Set rcParams suitable for a journal figure (serif, ~10 pt, 300 dpi)."""
    plt.rcParams.update({
        "font.family": "serif",
        # DejaVu Serif ships with matplotlib -> always available on the server
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "pdf.fonttype": 42,   # embed TrueType fonts (journal requirement)
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


# ----------------------------------------------------------------------------
# CSV loading / alignment
# ----------------------------------------------------------------------------
def _resolve_column(df, aliases, csv_path, kind):
    """Return the actual column name in df matching one of the aliases."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    raise ValueError(
        f"Could not find a {kind} column in {csv_path}. "
        f"Expected one of {aliases}, got columns {list(df.columns)}."
    )


def load_prediction_csv(csv_path, model_name):
    """Load one predictions CSV -> DataFrame indexed by timestamp with
    columns ['y_true', model_name]."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    ts_col = _resolve_column(df, TS_ALIASES, csv_path, "timestamp")
    yt_col = _resolve_column(df, TRUE_ALIASES, csv_path, "y_true")
    yp_col = _resolve_column(df, PRED_ALIASES, csv_path, "y_pred")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col]),
        "y_true": pd.to_numeric(df[yt_col], errors="coerce"),
        model_name: pd.to_numeric(df[yp_col], errors="coerce"),
    }).dropna(subset=["timestamp"])
    out = out.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return out


def align_csvs(csv_specs):
    """Inner-join all model CSVs on timestamp.

    csv_specs: list of (model_name, path).
    Returns a DataFrame indexed by timestamp with columns
    ['y_true', model_1, model_2, ...].
    """
    merged = None
    n_rows = {}
    for name, path in csv_specs:
        df = load_prediction_csv(path, name)
        n_rows[name] = len(df)
        if merged is None:
            merged = df
        else:
            # Consistency check on y_true across files (same season/site?)
            joined = merged.join(df, how="inner", rsuffix="__new")
            if "y_true__new" in joined.columns:
                diff = (joined["y_true"] - joined["y_true__new"]).abs().max()
                if pd.notna(diff) and diff > 1e-3:
                    warnings.warn(
                        f"[WARN] y_true in '{name}' differs from earlier CSVs "
                        f"(max |diff| = {diff:.4f} kW). Are all CSVs from the "
                        f"same season/site and inverse-transformed identically?"
                    )
                joined = joined.drop(columns=["y_true__new"])
            merged = joined

    if merged is None or merged.empty:
        raise ValueError("No overlapping timestamps across the given CSVs.")

    for name, n in n_rows.items():
        if n != len(merged):
            warnings.warn(
                f"[WARN] Inner join dropped rows for model '{name}': "
                f"{n} rows in file vs {len(merged)} aligned rows."
            )
    return merged


# ----------------------------------------------------------------------------
# Day statistics and representative-day selection
# ----------------------------------------------------------------------------
def compute_day_stats(merged, min_points, daylight_thresh_kw):
    """Per-calendar-day statistics on y_true.

    Returns a DataFrame indexed by date with columns:
    n_points, energy_kwh, peak_kw, vi, daylight_frac, complete, vi_valid.
    """
    g = merged["y_true"].groupby(merged.index.date)
    rows = []
    for day, s in g:
        s = s.sort_index()
        n = len(s)
        peak = float(s.max())
        energy = float(s.clip(lower=0).sum())  # hourly data -> kWh
        if peak > ZERO_MAX_GUARD_KW:
            vi = float(np.abs(np.diff(s.values)).sum() / peak)
            # NaN y_true values (not dropped at load time) make VI NaN;
            # mark such days invalid so they cannot be auto-selected.
            vi_valid = bool(np.isfinite(vi))
        else:
            vi = np.nan  # guard: zero-max (night-only / dead) day
            vi_valid = False
        daylight_frac = float((s > daylight_thresh_kw).mean())
        rows.append({
            "date": pd.Timestamp(day),
            "n_points": n,
            "energy_kwh": energy,
            "peak_kw": peak,
            "vi": vi,
            "daylight_frac": daylight_frac,
            "complete": n >= min_points,
            "vi_valid": vi_valid,
        })
    stats = pd.DataFrame(rows).set_index("date").sort_index()
    return stats


def select_representative_days(stats, high_energy_quantile):
    """Pick (clear, mid-VI, high-VI) days from the day-statistics table.

    Returns a list of (label, date, vi) ordered clear -> mid -> variable.
    """
    cand = stats[stats["complete"] & stats["vi_valid"]].copy()
    if cand.empty:
        raise ValueError("No complete days with a valid VI were found. "
                         "Check the input CSVs / --min-points.")

    picked = []          # list of (label, date, vi)
    used_dates = set()

    def take(label, sub):
        """Append the first not-yet-used date of `sub` (already ordered)."""
        for d in sub.index:
            if d not in used_dates:
                used_dates.add(d)
                picked.append((label, d, float(sub.loc[d, "vi"])))
                return True
        return False

    # 1) Highly variable (cloudy) day: highest VI.
    take("Highly variable day", cand.sort_values("vi", ascending=False))

    # 2) Clear-sky reference: lowest VI among high-energy days.
    thr = cand["energy_kwh"].quantile(high_energy_quantile)
    high_energy = cand[cand["energy_kwh"] >= thr]
    if high_energy.empty:
        high_energy = cand  # degenerate fallback (few days)
    take("Clear day", high_energy.sort_values("vi", ascending=True))

    # 3) Mid-VI day: closest to the median VI of the remaining candidates.
    med = cand["vi"].median()
    by_dist = cand.assign(_d=(cand["vi"] - med).abs()).sort_values("_d")
    take("Moderately variable day", by_dist)

    # Order panels clear -> moderate -> highly variable for the figure.
    order = {"Clear day": 0, "Moderately variable day": 1, "Highly variable day": 2}
    picked.sort(key=lambda t: order[t[0]])
    return picked


def resolve_manual_dates(dates_arg, stats):
    """Turn --dates 'YYYY-MM-DD,...' into the (label, date, vi) list."""
    picked = []
    for tok in dates_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        d = pd.Timestamp(tok).normalize()
        if d not in stats.index:
            raise ValueError(f"--dates: {tok} not present in the aligned data "
                             f"(available {stats.index.min().date()} .. "
                             f"{stats.index.max().date()}).")
        vi = float(stats.loc[d, "vi"]) if stats.loc[d, "vi_valid"] else float("nan")
        if not stats.loc[d, "complete"]:
            warnings.warn(f"[WARN] {tok} has only {int(stats.loc[d, 'n_points'])} "
                          f"points (incomplete day) -- plotting anyway.")
        picked.append(("Selected day", d, vi))
    if not picked:
        raise ValueError("--dates was given but no valid dates were parsed.")
    return picked


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_days(merged, picked, model_names, season, outdir):
    """One 1xN figure: actual (solid black + markers) vs model predictions
    (colored dashed lines) for each selected day."""
    n = len(picked)
    fig, axes = plt.subplots(
        1, n, figsize=(2.6 * n + 0.6, 2.9), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, (label, day, vi) in zip(axes, picked):
        day_mask = merged.index.normalize() == day
        sub = merged.loc[day_mask].sort_index()
        hours = sub.index.hour + sub.index.minute / 60.0

        # Actual power: solid black with circular markers
        ax.plot(hours, sub["y_true"].values, color="black", linestyle="-",
                marker="o", markerfacecolor="black", label="Actual", zorder=5)

        # Each model: distinct colored dashed line
        for k, m in enumerate(model_names):
            ax.plot(hours, sub[m].values,
                    color=MODEL_COLORS[k % len(MODEL_COLORS)],
                    linestyle=MODEL_LINESTYLES[k % len(MODEL_LINESTYLES)],
                    marker=MODEL_MARKERS[k % len(MODEL_MARKERS)],
                    markersize=2.8, markevery=2, label=m, zorder=4 - 0.1 * k)

        vi_txt = ""  # VI degenerates to 2.00 on most days at hourly resolution; omit from titles
        ax.set_title(f"{label} ({day.date()}){vi_txt}")
        ax.set_xlabel("Hour of day")
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 4))
        ax.grid(True, linestyle=":", linewidth=0.5)

    axes[0].set_ylabel("Active power (kW)")

    # Single shared legend above the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
               frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    pdf_path = os.path.join(outdir, f"representative_days_{season}.pdf")
    png_path = os.path.join(outdir, f"representative_days_{season}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def infer_season(csv_specs, season_arg):
    """Infer the season label from --season or from the CSV file names."""
    if season_arg:
        return season_arg
    pat = re.compile(r"(Spring|Summer|Autumn|Winter)", re.IGNORECASE)
    for _, path in csv_specs:
        m = pat.search(os.path.basename(path))
        if m:
            return m.group(1).capitalize()
    return "season"


def print_table(rows, headers):
    """Minimal fixed-width console table (no external deps)."""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Predicted-vs-actual plots for representative days "
                    "(clear / moderately variable / highly variable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv", action="append", required=True, metavar="NAME=PATH",
        help="Model predictions CSV as name=path "
             "(e.g. --csv Proposed=output/predictions_Autumn_agcf.csv). "
             "Repeat for multiple models; all must be the same season/site.",
    )
    p.add_argument("--dates", type=str, default=None,
                   help="Manual override: comma-separated YYYY-MM-DD dates "
                        "to plot instead of the automatic selection.")
    p.add_argument("--season", type=str, default=None,
                   help="Season label for output file names, e.g. Spring/"
                        "Summer/Autumn/Winter "
                        "(inferred from CSV file names if omitted).")
    p.add_argument("--min-points", type=int, default=MIN_POINTS_PER_DAY,
                   help="Minimum hourly points for a day to be 'complete'.")
    p.add_argument("--high-energy-quantile", type=float,
                   default=HIGH_ENERGY_QUANTILE,
                   help="Daily-energy quantile defining 'high-energy' days "
                        "for the clear-sky reference.")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                   help="Output directory for figures and CSV artifacts.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    # ---- Parse --csv name=path pairs -------------------------------------
    csv_specs = []
    for spec in args.csv:
        if "=" not in spec:
            raise SystemExit(f"--csv expects NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        csv_specs.append((name.strip(), os.path.expanduser(path.strip())))
    model_names = [name for name, _ in csv_specs]
    if len(set(model_names)) != len(model_names):
        raise SystemExit("Duplicate model names in --csv arguments.")

    season = infer_season(csv_specs, args.season)
    print(f"[INFO] Season label: {season}")
    print(f"[INFO] Models: {', '.join(model_names)}")

    # ---- Load and align ---------------------------------------------------
    merged = align_csvs(csv_specs)
    print(f"[INFO] Aligned rows: {len(merged)} "
          f"({merged.index.min()} .. {merged.index.max()})")

    # ---- Per-day statistics ------------------------------------------------
    daylight_thresh = DAYLIGHT_POWER_FRACTION * float(merged["y_true"].max())
    stats = compute_day_stats(merged, args.min_points, daylight_thresh)
    n_complete = int(stats["complete"].sum())
    print(f"[INFO] Calendar days in data: {len(stats)} "
          f"({n_complete} complete with >= {args.min_points} points)")

    # ---- Select representative days ----------------------------------------
    if args.dates:
        picked = resolve_manual_dates(args.dates, stats)
        print("[INFO] Using manually specified dates (--dates).")
    else:
        picked = select_representative_days(stats, args.high_energy_quantile)

    # ---- Report selection (for the manuscript text) ------------------------
    print("\nSelected representative days:")
    sel_rows = []
    for label, day, vi in picked:
        srow = stats.loc[day]
        sel_rows.append([
            label, str(day.date()),
            f"{vi:.3f}" if np.isfinite(vi) else "n/a",
            f"{srow['energy_kwh']:.1f}", f"{srow['peak_kw']:.1f}",
            f"{srow['daylight_frac']:.2f}", int(srow["n_points"]),
        ])
    print_table(sel_rows, ["role", "date", "VI", "energy_kWh",
                           "peak_kW", "daylight_frac", "n_points"])

    # ---- Plot ---------------------------------------------------------------
    pdf_path, png_path = plot_days(merged, picked, model_names, season,
                                   args.outdir)
    print(f"\n[INFO] Figure saved: {pdf_path}")
    print(f"[INFO] Figure saved: {png_path}")

    # ---- Artifacts: plotted data + full per-day stats -----------------------
    sel_dates = [day for _, day, _ in picked]
    data_out = merged.loc[merged.index.normalize().isin(sel_dates)].copy()
    data_csv = os.path.join(args.outdir,
                            f"representative_days_{season}_data.csv")
    data_out.to_csv(data_csv, index_label="timestamp")
    print(f"[INFO] Plotted series written: {data_csv}")

    stats_csv = os.path.join(args.outdir,
                             f"representative_days_{season}_selection.csv")
    stats_out = stats.copy()
    stats_out["selected_as"] = ""
    for label, day, _ in picked:
        stats_out.loc[day, "selected_as"] = label
    stats_out.to_csv(stats_csv, index_label="date")
    print(f"[INFO] Per-day statistics written: {stats_csv}")


if __name__ == "__main__":
    main()
