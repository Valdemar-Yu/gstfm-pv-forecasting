#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_predictions.py
===================
Bridge between saved seasonal checkpoints and the revision analysis scripts.

For one (site, season) pair it:
  1) loads the seasonal CSV and reproduces the EXACT training split via
     data.split_data_cnn(df, 0.8, 0.1, lookback=24);
  2) builds the test Dataset/DataLoader (data_detime, lookback=24,
     lookforward=--pred-len, multi_steps=(pred_len>1), batch=128,
     shuffle=False) -- identical to the training drivers;
  3) loads a checkpoint robustly (full-model pickle first; falls back to
     state_dict + fresh instantiation selected with --model);
  4) runs inference, inverse-transforms predictions AND ground truth back
     to kW with the StandardScaler (scalar_y) returned by split_data_cnn;
  5) aligns every prediction with its wall-clock timestamp (see the
     "TIMESTAMP ALIGNMENT" comment below for the derivation);
  6) writes results/predictions_{site}_{season}_{modelname}.csv with columns
     timestamp,y_true_kw,y_pred_kw and prints MAE/RMSE/R2/MBE
     (utils.tools.metrics_of_pv) as a sanity check against the paper tables.

Run from anywhere (paths are resolved relative to this file / --repo), e.g.
    python scripts/dump_predictions.py --season Autumn --model agcf
    python scripts/dump_predictions.py --season Winter --model lstm \
        --ckpt results/ckpt/lstm_7-First-Solar_Winter_h1_s42.pt
    python scripts/dump_predictions.py --site 1B --season Summer --model dlinear

Checkpoint convention (shared contract with train_baseline.py / train_gstfm.py):
  * Default checkpoint path for EVERY model:
        <repo>/results/ckpt/{model}_{site}_{season}_h{pred_len}_s{seed}.pt
    - train_gstfm.py saves a plain state_dict under model tag 'gstfm'
      (plus a JSON args sidecar <stem>.json, which this script reads to
      re-instantiate the exact architecture);
    - train_baseline.py's EarlyStopping saves a FULL model pickle
      (torch.save(model, ...) with pickle_module=dill) — also supported,
      as are {'state_dict': ...}-style wrapper dicts.
  * Default hyperparameters used to re-instantiate models from a state_dict
    follow THIS repo's training drivers (overridden by the sidecar if found):
      - proposed iTransformer-LSTM (train_baseline.py):
        dim_embed=128, dim_lstm=128, depth=4 (iTransformer), heads=6,
        depth_lstm=3, length_input=24, length_pre=pred_len, input_size=5
      - agcf / gstfm variant (train_gstfm.py defaults): same backbone,
        use_agcf=True, gate_width=6, gate_depth=3, gate_residual=True,
        gate_channel=False
      - LSTM baseline (train_baseline.py): hidden=128, layers=3, input=5
      - iTransformer baseline (train_baseline.py): iTransformer_single,
        num_variates=5, lookback_len=24, dim=256, depth=5,
        heads=6, num_tokens_per_variate=1, use_reversible_instance_norm=True
      - dlinear / timesnet / fedformer: models.baselines.build_baseline(name,
        seq_len=24, pred_len=pred_len, enc_in=5) (shared factory contract)
"""

import argparse
import json
import os
import sys
import csv
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Configurable constants (paper defaults) --------------------------------
# ---------------------------------------------------------------------------
LOOKBACK = 24          # time_length in the training scripts (24 h)
BATCH_SIZE = 128
INPUT_SIZE = 5         # AP + 4 covariates
TRAIN_RATIO, TEST_RATIO = 0.8, 0.1  # split_data_cnn(df, 0.8, 0.1, lookback)
SEASONS = ("Spring", "Summer", "Autumn", "Winter")
SITES = ("7-First-Solar", "1B")
# Seasonal CSV filename year span differs per site:
#   data/7-First-Solar/{Season}_7-First-Solar_2019_2022_h.csv
#   data/1B/{Season}_1B_2019_2023_h.csv
SITE_YEARS = {"7-First-Solar": "2019_2022", "1B": "2019_2023"}
MODEL_CHOICES = ["itransformer_lstm", "agcf", "gstfm", "lstm", "itransformer",
                 "dlinear", "timesnet", "fedformer"]

# Repo root = parent directory of scripts/ (this file lives in <repo>/scripts).
# Override with --repo on the server if the layout differs.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.dirname(_THIS_DIR)
OUTPUT_DIR = os.path.join(DEFAULT_REPO, "results")


def resolve_repo(repo_arg):
    """Find the repo root; accept --repo, the default parent folder, or cwd."""
    candidates = [repo_arg, DEFAULT_REPO, os.getcwd()]
    for cand in candidates:
        if cand and os.path.isdir(os.path.join(cand, "data")) \
                and os.path.isdir(os.path.join(cand, "models")) \
                and os.path.isdir(os.path.join(cand, "utils")):
            return os.path.abspath(cand)
    raise FileNotFoundError(
        "Could not locate the repo root (needs data/, models/, utils/). "
        "Pass it explicitly with --repo /path/to/GSTFM-revision")


def build_model(model_key, torch, pred_len=1, hparams=None):
    """Instantiate a model with this repo's training-driver hyperparameters.

    Only used when the checkpoint turns out to be a state_dict.
    Imports are done lazily so that e.g. the agcf module (which may need
    Python >= 3.10 for its type hints / optional pennylane) is only touched
    when actually requested.

    NOTE(review): the hybrid backbone defaults below follow THIS repo's
    trainers (train_gstfm.py / train_baseline.py: dim_embed=128, dim_lstm=128,
    depth=4, heads=6, depth_lstm=3), NOT the old repo's Optuna config
    (reference/pre_iTransformer_LSTM_parallel.py: 32/32/heads=12) — a
    state_dict saved by train_gstfm.py would not load into the old shapes.
    `hparams` (the JSON sidecar train_gstfm.py writes next to each ckpt)
    overrides these defaults when available.
    """
    hp = hparams or {}
    if model_key == "itransformer_lstm":
        # Proposed base model (train_baseline.py build_model config)
        from models.iTransformer_LSTM import iTransformer_LSTM
        return iTransformer_LSTM(input_size=INPUT_SIZE, length_pre=pred_len,
                                 dim_lstm=hp.get("dim_lstm", 128),
                                 depth_lstm=hp.get("depth_lstm", 3),
                                 length_input=LOOKBACK,
                                 dim_embed=hp.get("dim_embed", 128),
                                 depth=hp.get("depth", 4),
                                 heads=hp.get("heads", 6))
    if model_key in ("agcf", "gstfm"):
        # Gated-fusion variant ('gstfm' is train_gstfm.py's checkpoint tag):
        # same backbone + classical AGCF unit enabled (train_gstfm.py defaults).
        from models.iTransformer_LSTM_agcf import iTransformer_LSTM as QGatedModel
        return QGatedModel(input_size=INPUT_SIZE, length_pre=pred_len,
                           dim_lstm=hp.get("dim_lstm", 128),
                           depth_lstm=hp.get("depth_lstm", 3),
                           length_input=LOOKBACK,
                           dim_embed=hp.get("dim_embed", 128),
                           depth=hp.get("depth", 4),
                           heads=hp.get("heads", 6),
                           use_agcf=hp.get("use_agcf", True),
                           # gate_type / head are what distinguish the Table-3
                           # ablation variants. Omitting them here would rebuild
                           # a KAN head for an 'mlp_head' checkpoint (and a AGCF
                           # gate for an 'se_gate' one); load_state_dict(strict=
                           # False) would then silently leave them random.
                           gate_type=hp.get("gate_type"),
                           head=hp.get("head", "kan"),
                           gate_width=hp.get("gate_width", 6),
                           gate_depth=hp.get("gate_depth", 3),
                           gate_residual=hp.get("gate_residual", True),
                           gate_channel=hp.get("gate_channel", False))
    if model_key == "lstm":
        from models.LSTM import LSTM
        return LSTM(input_size=INPUT_SIZE, hidden_size=128, num_layers=3,
                    output_size=pred_len)
    if model_key == "itransformer":
        from models.iTransformer import iTransformer_single
        return iTransformer_single(num_variates=INPUT_SIZE, lookback_len=LOOKBACK,
                                   pred_length=pred_len, dim=256, depth=5, heads=6,
                                   num_tokens_per_variate=1,
                                   use_reversible_instance_norm=True)
    if model_key in ("dlinear", "timesnet", "fedformer"):
        # Reference baselines via the shared factory (models/baselines):
        # forward(FloatTensor[B, seq_len, enc_in]) -> FloatTensor[B, pred_len].
        from models.baselines import build_baseline
        return build_baseline(model_key, seq_len=LOOKBACK, pred_len=pred_len,
                              enc_in=INPUT_SIZE)
    raise ValueError(f"Unknown --model '{model_key}'. "
                     f"Choose from {', '.join(MODEL_CHOICES)}.")


def robust_torch_load(path, device, torch):
    """torch.load that works across PyTorch versions and pickle modules.

    Checkpoints in this repo were saved with pickle_module=dill; newer
    PyTorch (>= 2.6) defaults to weights_only=True which rejects full-model
    pickles, so we explicitly request weights_only=False when supported.
    """
    try:
        import dill
        pickle_module = dill
    except ImportError:
        pickle_module = None  # fall back to std pickle
    kwargs = {"map_location": device}
    if pickle_module is not None:
        kwargs["pickle_module"] = pickle_module
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        # Older PyTorch without the weights_only kwarg
        return torch.load(path, **kwargs)


def load_sidecar_hparams(ckpt_path):
    """Read the JSON args sidecar train_gstfm.py writes next to each ckpt
    (<stem>.json); returns {} when absent/unreadable."""
    sidecar = os.path.splitext(ckpt_path)[0] + ".json"
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                hp = json.load(f)
            print(f"[load] hyperparameters from sidecar {sidecar}")
            return hp if isinstance(hp, dict) else {}
        except (OSError, ValueError) as exc:
            print(f"[load] ignoring unreadable sidecar {sidecar}: {exc}")
    return {}


def load_model_from_checkpoint(ckpt_path, model_key, device, torch, pred_len=1):
    """Load a checkpoint that may be a full nn.Module pickle, a raw
    state_dict, or a wrapper dict. Returns (model, inferred_name)."""
    obj = robust_torch_load(ckpt_path, device, torch)

    # Case 1: full-model save (torch.save(model, path)) -- the repo default.
    if isinstance(obj, torch.nn.Module):
        print("[load] Checkpoint is a full nn.Module pickle "
              f"({obj.__class__.__name__}).")
        return obj.to(device)

    # Case 2: dict-like. Could be a raw state_dict or a wrapper such as
    # {'state_dict': ...} / {'model_state_dict': ...} / {'model': nn.Module}.
    if isinstance(obj, (dict, OrderedDict)):
        state = obj
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in state:
                inner = state[key]
                if isinstance(inner, torch.nn.Module):
                    print(f"[load] Checkpoint dict wraps a full model under '{key}'.")
                    return inner.to(device)
                if isinstance(inner, (dict, OrderedDict)):
                    print(f"[load] Using nested state_dict under key '{key}'.")
                    state = inner
                break
        if model_key is None:
            raise SystemExit(
                "[error] Checkpoint is a state_dict; you must pass "
                f"--model {{{','.join(MODEL_CHOICES)}}} so the "
                "right architecture can be instantiated.")
        model = build_model(model_key, torch, pred_len=pred_len,
                            hparams=load_sidecar_hparams(ckpt_path)).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[load] load_state_dict(strict=False): "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys.")
        if missing:
            print("        missing   :", list(missing))
        if unexpected:
            print("        unexpected:", list(unexpected))
        # Checkpoints written before agcf.proj_in was made eager (it used to be
        # built lazily on the first forward) put its weights in the 'unexpected'
        # bucket. Materialise and reload so those old checkpoints still restore.
        if any("proj_in" in k for k in unexpected):
            print("[load] Legacy checkpoint with lazily-built submodules; "
                  "materialising with a dummy forward and re-loading ...")
            with torch.no_grad():
                model(torch.zeros(2, LOOKBACK, INPUT_SIZE, device=device))
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[load] second pass: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys.")
        if missing:
            raise SystemExit(
                f"[error] {len(missing)} parameters in the rebuilt model were not "
                f"found in the checkpoint, e.g. {list(missing)[:5]}.\n"
                f"        They would stay at their random initialisation and the "
                f"dumped predictions would be meaningless.\n"
                f"        The architecture rebuilt from the sidecar does not match "
                f"the checkpoint — check gate_type/head/dim_* in "
                f"{os.path.splitext(ckpt_path)[0]}.json")
        return model
    raise SystemExit(f"[error] Unsupported checkpoint object type: {type(obj)}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump per-timestamp test-set predictions (kW) for a saved "
                    "checkpoint so downstream revision analyses can use them.")
    parser.add_argument("--season", default="Autumn", choices=SEASONS,
                        help="Season split to evaluate (default: Autumn)")
    parser.add_argument("--site", default="7-First-Solar", choices=SITES,
                        help="PV site / dataset folder under data/ "
                             "(default: 7-First-Solar; 1B uses the "
                             "2019_2023 filename pattern).")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path. Default (shared contract): "
                             "<repo>/results/ckpt/"
                             "{model}_{site}_{season}_h{pred_len}_s{seed}.pt "
                             "(requires --model when omitted).")
    parser.add_argument("--model", default=None, choices=MODEL_CHOICES,
                        help="Architecture to instantiate IF the checkpoint is "
                             "a state_dict (full-model pickles ignore this). "
                             "Also used for the default ckpt path and to name "
                             "the output CSV when given.")
    parser.add_argument("--pred-len", type=int, default=1,
                        help="Forecast horizon (default: 1). Must match the "
                             "checkpoint's training horizon; only the first "
                             "step (t+1) is dumped to the CSV.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed tag in the default checkpoint filename "
                             "(default: 42).")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help="Repo root (default: parent directory of scripts/).")
    parser.add_argument("--out_dir", default=OUTPUT_DIR,
                        help="Directory for the predictions CSV "
                             "(default: <repo>/results).")
    args = parser.parse_args()

    repo = resolve_repo(args.repo)
    # Repo modules ('models', 'data', 'utils') import each other via absolute
    # imports rooted at the repo, so the repo root must be on sys.path. This
    # is also required to unpickle full-model checkpoints.
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from data import split_data_cnn, data_detime
    from utils.tools import metrics_of_pv, same_seeds

    same_seeds(args.seed)  # same seed as the training scripts (inference-only anyway)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] repo={repo}\n[setup] device={device}")

    if args.ckpt is None:
        if args.model is None:
            raise SystemExit("[error] Without --ckpt you must pass --model so "
                             "the default checkpoint path can be built.")
        # Shared naming contract with train_baseline.py / the model drivers.
        # train_gstfm.py names its checkpoints after MODEL_NAME ('GSTFM',
        # uppercase), not after the lowercase CLI key — and the server FS is
        # case-sensitive even though macOS is not.
        stem = "GSTFM" if args.model in ("gstfm", "agcf") else args.model
        ckpt_path = os.path.join(
            repo, "results", "ckpt",
            f"{stem}_{args.site}_{args.season}"
            f"_h{args.pred_len}_s{args.seed}.pt")
    else:
        ckpt_path = args.ckpt
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"[error] Checkpoint not found: {ckpt_path}")
    print(f"[setup] checkpoint={ckpt_path}")

    # ------------------------------------------------------------------
    # 1) Load CSV and reproduce the training-time split exactly.
    # ------------------------------------------------------------------
    csv_path = os.path.join(repo, "data", args.site,
                            f"{args.season}_{args.site}_"
                            f"{SITE_YEARS[args.site]}_h.csv")
    df_all = pd.read_csv(csv_path, header=0)
    print(f"[data] {csv_path} -> {len(df_all)} rows, columns={list(df_all.columns)}")

    (data_train, data_valid, data_test,
     ts_train, ts_valid, ts_test, scalar_y) = split_data_cnn(
        df_all, TRAIN_RATIO, TEST_RATIO, LOOKBACK)

    # ------------------------------------------------------------------
    # 2) Test Dataset / DataLoader exactly as during training evaluation.
    # ------------------------------------------------------------------
    multi_steps = args.pred_len > 1  # same switch as the training drivers
    dataset_test = data_detime(data=data_test, lookback_length=LOOKBACK,
                               lookforward_length=args.pred_len,
                               multi_steps=multi_steps)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    n_samples = len(dataset_test)

    # ------------------------------------------------------------------
    # TIMESTAMP ALIGNMENT (the subtle part) ------------------------------
    # split_data_cnn slices BOTH the scaled data and the timestamps with the
    # SAME indices:
    #     data_test = data[num_train + num_valid - lookback : length]
    #     ts_test   = timestamp[num_train + num_valid - lookback : length]
    # so row j of data_test corresponds to row j of ts_test; both include
    # `lookback` warm-up rows taken from BEFORE the test split boundary.
    #
    # data_detime builds sample i as
    #     x = data_test[i : i + lookback]                          (inputs)
    #     y = data_test[i + lookback : i + lookback + pred_len, 0] (multi_steps)
    #     y = data_test[i + lookback + pred_len - 1, 0]            (single-step)
    # We always dump the FIRST forecast step (t+1). With multi_steps=True
    # (pred_len > 1) that step sits at row i + lookback; with pred_len = 1
    # (multi_steps=False) the single target also sits at i + lookback.
    # Hence sample i's dumped prediction has wall-clock time
    #     ts_test.iloc[i + lookback]
    # With lookback=24 the first predicted row is index 24, i.e. exactly the
    # first row of the "real" test split (the warm-up rows are only ever
    # used as inputs). Number of samples:
    #     N = len(data_test) - lookback - pred_len + 1.
    # ------------------------------------------------------------------
    first = LOOKBACK
    ts_target = pd.to_datetime(
        ts_test["date"].iloc[first: first + n_samples]).reset_index(drop=True)
    assert len(ts_target) == n_samples, (
        f"Timestamp alignment failed: {len(ts_target)} timestamps vs "
        f"{n_samples} test samples")

    # ------------------------------------------------------------------
    # 3) Load the checkpoint (full model or state_dict).
    # ------------------------------------------------------------------
    model = load_model_from_checkpoint(ckpt_path, args.model, device, torch,
                                       pred_len=args.pred_len)
    model.eval()

    # ------------------------------------------------------------------
    # 4) Inference on the test set (order preserved: shuffle=False).
    # ------------------------------------------------------------------
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)
            y_pre = model(x)
            # All models return [B, pred_len] ([B, 1] for single-step);
            # keep only the FIRST forecast step (t+1) for the dump and
            # reshape defensively in case of an extra singleton dim.
            preds.append(y_pre.reshape(len(x), -1)[:, :1].cpu().numpy())
            trues.append(y.reshape(len(x), -1)[:, :1].numpy())
    preds = np.concatenate(preds, axis=0)  # [N, 1], standardized units
    trues = np.concatenate(trues, axis=0)  # [N, 1], standardized units

    # Inverse-transform back to kW with the StandardScaler fitted on the
    # (clipped-at-0) TRAIN target, exactly like utils.tools.evaluate does.
    preds_kw = scalar_y.inverse_transform(preds).ravel()
    trues_kw = scalar_y.inverse_transform(trues).ravel()

    # ------------------------------------------------------------------
    # 5)+6) Write CSV and print metrics sanity check.
    # ------------------------------------------------------------------
    if args.model is not None:
        model_name = args.model
    else:
        # Derive a name from the checkpoint filename, dropping the season tag.
        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        model_name = model_name.replace(f"_{args.season}", "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir,
                           f"predictions_{args.site}_{args.season}_{model_name}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "y_true_kw", "y_pred_kw"])
        for t, yt, yp in zip(ts_target, trues_kw, preds_kw):
            writer.writerow([t.strftime("%Y-%m-%d %H:%M:%S"),
                             f"{yt:.6f}", f"{yp:.6f}"])
    print(f"[out] wrote {n_samples} rows -> {out_csv}")

    # Metrics in kW — should reproduce the paper-table numbers for this
    # season/model (same pipeline as utils.tools.evaluate with scalar).
    mae, rmse, r2, mbe = metrics_of_pv(preds_kw, trues_kw)
    print("\nSanity-check metrics on the test set (kW, inverse-transformed):")
    header = f"{'Season':<8}{'Model':<36}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'MBE':>10}"
    print(header)
    print("-" * len(header))
    print(f"{args.season:<8}{model_name:<36}{mae:>10.4f}{rmse:>10.4f}"
          f"{r2:>10.4f}{mbe:>10.4f}")


if __name__ == "__main__":
    main()
