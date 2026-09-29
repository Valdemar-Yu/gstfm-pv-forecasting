#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
param_latency.py
================
Reviewer 2, item 5: "Add parameter counts and inference latency for GSTFM and
all baselines."

For every model instantiable from this repository this script reports:
  1) trainable parameter count (sum of p.numel() for p.requires_grad),
  2) inference latency for input torch.randn(B, LOOKBACK, N_FEATURES) at
     B=1 and B=32 (20 warm-up iterations, 100 timed iterations,
     torch.cuda.synchronize() around each timed call on GPU), mean +/- std in ms,
  3) peak GPU memory allocated during the inference runs (MB, CUDA only),
  4) a Persistence baseline (y_hat = x[:, -1, 0], zero parameters).

Models covered (constructor signatures verified against this repo):
  * Persistence                    (trivial, defined below)
  * LSTM                           models/LSTM.py
  * TCN                            models/TCN.py
  * iTransformer                   models/iTransformer/iTransformer.py (iTransformer_single)
  * iTransformer-LSTM (base)       models/iTransformer_LSTM.py
  * GSTFM (gated, use_agcf=True)    models/iTransformer_LSTM_agcf.py
  * DLinear / TimesNet / FEDformer models/baselines (build_baseline factory;
                                   wrappers around the Time-Series-Library
                                   implementations, forward: [B,L,C]->[B,pred_len])

Hyper-parameters follow the training drivers in THIS repo:
  - proposed model (train_gstfm.py / train_baseline.py defaults):
      input_size=5, length_input=24, length_pre=1,
      dim_embed=128, dim_lstm=128, depth=4, heads=6, depth_lstm=3
  - standalone iTransformer (train_baseline.py):
      num_variates=5, lookback_len=24, pred_length=1, dim=256, depth=5, heads=6
  - standalone LSTM (train_baseline.py): hidden_size=128, num_layers=3, output_size=1
  - TCN (pre_TCN.py): num_channels=[32]*3, kernel_size=3; NOTE: the TCN head is
      nn.Linear(num_channels[-1]*lenth_back, 20), so lenth_back MUST equal the
      lookback length -> we pass lenth_back=LOOKBACK (=24) here, whereas
      pre_TCN.py trains with a 48-step lookback. Adjust if the paper's TCN
      config differs.

Notes / assumptions
-------------------
* Latency and parameter counts do not depend on trained weights, so models are
  benchmarked with random initialization; no checkpoint loading is required.
* The AGCF (classical AGCF unit) lazily creates its input projection
  (`proj_in`) on the first forward call, therefore ONE forward pass is run
  before counting parameters (done uniformly for all models).
* If PennyLane is installed on the server, the AGCF falls back to a per-sample
  Python loop over the qnode -- the measured B=32 latency then honestly
  includes that loop. Without PennyLane a classical surrogate MLP is used.
* Run from the repo root (or anywhere): the repo root (parent directory of
  scripts/) is added to sys.path automatically.
* GPU note: the paper reports timings measured on an NVIDIA GeForce RTX 3090;
  re-run this script on that machine so the table matches the manuscript.

Usage (on the authors' Linux server with PyTorch):
    python scripts/param_latency.py
    python scripts/param_latency.py --device cpu --iters 200
"""

import argparse
import csv
import os
import statistics
import sys
import time

# --------------------------------------------------------------------------
# Path setup: make the repo packages (models/, utils/, data/) importable.
# This script lives in <repo_root>/scripts/, so the repo root is the parent
# directory of the script's directory -- override with --repo-root if needed.
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT_DIR = os.path.join(DEFAULT_REPO_ROOT, "results")


def parse_args():
    p = argparse.ArgumentParser(
        description="Parameter counts, inference latency and peak GPU memory "
                    "for GSTFM and baselines (Reviewer 2, item 5).")
    p.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT,
                   help="Repo root (default: parent directory of scripts/).")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                   help="Directory for the CSV artifact (default: <repo>/results).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Device for benchmarking (default: cuda if available).")
    p.add_argument("--batch-sizes", type=str, default="1,32",
                   help="Comma-separated batch sizes to time (default: 1,32).")
    p.add_argument("--warmup", type=int, default=20,
                   help="Warm-up iterations before timing (default: 20).")
    p.add_argument("--iters", type=int, default=100,
                   help="Timed iterations (default: 100).")
    p.add_argument("--lookback", type=int, default=24,
                   help="Input window length (default: 24).")
    p.add_argument("--features", type=int, default=5,
                   help="Number of input features incl. target (default: 5).")
    p.add_argument("--pred-len", type=int, default=1,
                   help="Prediction horizon (default: 1).")
    p.add_argument("--seed", type=int, default=3407, help="Random seed.")
    return p.parse_args()


ARGS = parse_args()
sys.path.insert(0, ARGS.repo_root)

import torch                    # noqa: E402
import torch.nn as nn           # noqa: E402


# ==========================================================================
# Persistence baseline: y_hat(t+1) = y(t)  (last observed target value).
# Zero trainable parameters; we still time the tensor op for completeness.
# ==========================================================================
class Persistence(nn.Module):
    """Naive persistence forecaster on the first (target) channel."""

    def forward(self, x):
        # x: [B, T, C] -> [B, 1]
        return x[:, -1, 0:1]


# ==========================================================================
# Model builders. Each returns a fresh nn.Module on CPU.
# Constructor signatures were verified against the repo source files.
# ==========================================================================
def build_persistence(cfg):
    return Persistence()


def build_lstm(cfg):
    """Standalone LSTM baseline -- hyper-parameters from pre_lstm.py."""
    from models.LSTM import LSTM
    return LSTM(input_size=cfg["features"], hidden_size=128, num_layers=3,
                output_size=cfg["pred_len"])


def build_tcn(cfg):
    """TCN baseline -- channels/kernel from pre_TCN.py.

    NOTE: `lenth_back` (sic, repo spelling) must equal the lookback length
    because the head flattens num_channels[-1] * lenth_back features.
    """
    from models.TCN import TCN
    return TCN(input_size=cfg["features"], output_size=cfg["pred_len"],
               num_channels=[32] * 3, kernel_size=3, lenth_back=cfg["lookback"])


def build_itransformer(cfg):
    """Standalone iTransformer baseline -- hyper-parameters from pre_iTransformer.py."""
    from models.iTransformer import iTransformer_single
    return iTransformer_single(num_variates=cfg["features"],
                               lookback_len=cfg["lookback"],
                               pred_length=cfg["pred_len"],
                               dim=256, depth=5, heads=6,
                               num_tokens_per_variate=1,
                               use_reversible_instance_norm=True)


# Shared hyper-parameters of the proposed hybrid model.
# NOTE(review): aligned with THIS repo's training drivers (train_gstfm.py /
# train_baseline.py: dim_embed=128, dim_lstm=128, depth=4, heads=6,
# depth_lstm=3) so the parameter/latency table describes the models actually
# trained in the revision. The old repo's Optuna config
# (reference/pre_iTransformer_LSTM_parallel.py) was 32/32/heads=12.
HYBRID_HP = dict(dim_embed=128, dim_lstm=128, depth=4, heads=6, depth_lstm=3)


def build_itransformer_lstm(cfg):
    """iTransformer-LSTM base (no gating) -- models/iTransformer_LSTM.py."""
    from models.iTransformer_LSTM import iTransformer_LSTM
    return iTransformer_LSTM(input_size=cfg["features"],
                             length_pre=cfg["pred_len"],
                             length_input=cfg["lookback"],
                             **HYBRID_HP)


def build_qstfm_gated(cfg):
    """GSTFM / gated-fusion model -- models/iTransformer_LSTM_agcf.py with use_agcf=True.

    Uses the same backbone hyper-parameters as the base hybrid; gating knobs
    at their module defaults (gate_width=6, gate_depth=3, gate_residual=True).
    """
    from models.iTransformer_LSTM_agcf import iTransformer_LSTM as QGatedModel
    return QGatedModel(input_size=cfg["features"],
                       length_pre=cfg["pred_len"],
                       length_input=cfg["lookback"],
                       use_agcf=True,
                       **HYBRID_HP)


# --------------------------------------------------------------------------
# Reference baselines from models/baselines (Time-Series-Library wrappers).
# The shared factory returns a module whose forward maps
# FloatTensor[B, seq_len, enc_in] -> FloatTensor[B, pred_len]; any decoder
# inputs / time-mark tensors are handled internally by the wrapper, so these
# baselines are benchmarked with the exact same harness as all other models.
# --------------------------------------------------------------------------
def build_dlinear(cfg):
    """DLinear baseline via the shared factory (models/baselines)."""
    from models.baselines import build_baseline
    return build_baseline("dlinear", seq_len=cfg["lookback"],
                          pred_len=cfg["pred_len"], enc_in=cfg["features"])


def build_timesnet(cfg):
    """TimesNet baseline via the shared factory (models/baselines)."""
    from models.baselines import build_baseline
    return build_baseline("timesnet", seq_len=cfg["lookback"],
                          pred_len=cfg["pred_len"], enc_in=cfg["features"])


def build_fedformer(cfg):
    """FEDformer baseline via the shared factory (models/baselines)."""
    from models.baselines import build_baseline
    return build_baseline("fedformer", seq_len=cfg["lookback"],
                          pred_len=cfg["pred_len"], enc_in=cfg["features"])


# MODELS registry: (display name, builder). Order = row order in the table.
MODELS = [
    ("Persistence", build_persistence),
    ("LSTM", build_lstm),
    ("TCN", build_tcn),
    ("iTransformer", build_itransformer),
    ("iTransformer-LSTM", build_itransformer_lstm),
    ("GSTFM (gated)", build_qstfm_gated),
    ("DLinear", build_dlinear),
    ("TimesNet", build_timesnet),
    ("FEDformer", build_fedformer),
]


# ==========================================================================
# Measurement helpers
# ==========================================================================
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_forward(model, x, device, warmup, iters):
    """Return (mean_ms, std_ms) over `iters` timed forward passes."""
    use_cuda = (device.type == "cuda")
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if use_cuda:
            torch.cuda.synchronize(device)
        times_ms = []
        for _ in range(iters):
            if use_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x)
            if use_cuda:
                torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = statistics.fmean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return mean_ms, std_ms


def benchmark_model(name, builder, cfg, device):
    """Build + benchmark one model; never raises (errors captured in the row)."""
    row = {
        "Model": name,
        "Params": None,
        "Params_M": None,
        "PeakMem_MB": None,
        "Error": "",
    }
    for b in cfg["batch_sizes"]:
        row[f"Latency_B{b}_mean_ms"] = None
        row[f"Latency_B{b}_std_ms"] = None

    try:
        model = builder(cfg)
        model = model.to(device)
        model.eval()

        # One materialization forward BEFORE counting parameters: the AGCF
        # creates its `proj_in` Linear lazily on the first call.
        with torch.no_grad():
            model(torch.randn(1, cfg["lookback"], cfg["features"], device=device))

        row["Params"] = count_trainable_params(model)
        row["Params_M"] = row["Params"] / 1e6

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        for b in cfg["batch_sizes"]:
            x = torch.randn(b, cfg["lookback"], cfg["features"], device=device)
            mean_ms, std_ms = time_forward(model, x, device,
                                           cfg["warmup"], cfg["iters"])
            row[f"Latency_B{b}_mean_ms"] = mean_ms
            row[f"Latency_B{b}_std_ms"] = std_ms

        if device.type == "cuda":
            # Peak over ALL inference runs above (dominated by the largest B).
            row["PeakMem_MB"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # keep going, report the failure in the table
        row["Error"] = f"{type(exc).__name__}: {exc}"
    return row


# ==========================================================================
# Output formatting
# ==========================================================================
def fmt_int(v):
    return f"{v:,}" if v is not None else "-"


def fmt_float(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "-"


def fmt_latency(mean, std):
    if mean is None:
        return "-"
    return f"{mean:.3f} +/- {std:.3f}"


def print_table(rows, batch_sizes, device):
    headers = ["Model", "Params", "Params(M)"]
    headers += [f"Latency@B{b}(ms)" for b in batch_sizes]
    headers += ["PeakMem(MB)", "Error"]

    table = []
    for r in rows:
        line = [r["Model"], fmt_int(r["Params"]), fmt_float(r["Params_M"], 4)]
        for b in batch_sizes:
            line.append(fmt_latency(r[f"Latency_B{b}_mean_ms"],
                                    r[f"Latency_B{b}_std_ms"]))
        line.append(fmt_float(r["PeakMem_MB"], 2) if device.type == "cuda" else "N/A(CPU)")
        line.append(r["Error"][:60] if r["Error"] else "")
        table.append(line)

    widths = [max(len(h), max((len(t[i]) for t in table), default=0))
              for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for line in table:
        print(" | ".join(c.ljust(w) for c, w in zip(line, widths)))


def write_csv(rows, batch_sizes, device, out_path):
    fields = ["Model", "Params", "Params_M"]
    for b in batch_sizes:
        fields += [f"Latency_B{b}_mean_ms", f"Latency_B{b}_std_ms"]
    fields += ["PeakMem_MB", "Device", "Error"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields}
            out["Device"] = str(device)
            writer.writerow(out)


# ==========================================================================
# Main
# ==========================================================================
def main():
    torch.manual_seed(ARGS.seed)

    if ARGS.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(ARGS.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(ARGS.seed)

    batch_sizes = [int(b) for b in ARGS.batch_sizes.split(",") if b.strip()]
    cfg = {
        "lookback": ARGS.lookback,
        "features": ARGS.features,
        "pred_len": ARGS.pred_len,
        "batch_sizes": batch_sizes,
        "warmup": ARGS.warmup,
        "iters": ARGS.iters,
    }

    print(f"# param_latency.py | device={device} | input=[B,{cfg['lookback']},{cfg['features']}]"
          f" | warmup={cfg['warmup']} | iters={cfg['iters']}")
    if device.type == "cuda":
        print(f"# GPU: {torch.cuda.get_device_name(device)}")
    print("# NOTE: the paper reports timings measured on an NVIDIA GeForce RTX 3090.")
    print(f"# torch {torch.__version__}\n")

    rows = []
    for name, builder in MODELS:
        print(f"[bench] {name} ...")
        rows.append(benchmark_model(name, builder, cfg, device))

    print()
    print_table(rows, batch_sizes, device)

    os.makedirs(ARGS.out_dir, exist_ok=True)
    csv_path = os.path.join(ARGS.out_dir, "param_latency.csv")
    write_csv(rows, batch_sizes, device, csv_path)
    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()
