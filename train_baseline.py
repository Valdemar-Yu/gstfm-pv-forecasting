"""Generic baseline trainer for the GSTFM revision experiments.

Exposes main(args) -> dict (importable by the sweep drivers) and an argparse CLI.

Model registry
--------------
- 'dlinear' / 'timesnet' / 'fedformer' : via models.baselines.build_baseline
  (shared contract: forward(x[B, seq_len, enc_in]) -> [B, pred_len])
- 'itransformer_lstm'                  : models.iTransformer_LSTM.iTransformer_LSTM
- 'lstm'                               : models.LSTM.LSTM
- 'itransformer'                       : models.iTransformer.iTransformer_single
- 'persistence'                        : no training, y_hat = x[:, -1, 0] repeated

Training protocol (paper Table 1): Adam lr 1e-4, batch 128, up to 150 epochs,
EarlyStopping(patience=10, delta=1e-4) on the validation loss, seed 42 default,
lookback seq_len = 24, split 0.8/0.1, MAE criterion (nn.L1Loss reduction='sum').

Metrics are always recorded at BOTH scales:
- 'std' : on standardized values (paper's reported values / 100)
- 'kw'  : inverse-transformed to kW with scalar_y

Results CSV columns:
site,season,model,pred_len,seed,scale,mae,rmse,r2,mbe,epochs_run,train_seconds
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
import warnings

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data import split_data_cnn, data_detime  # noqa: E402
from utils.tools import metrics_of_pv, EarlyStopping, same_seeds, train, evaluate  # noqa: E402

MODEL_NAMES = [
    "dlinear",
    "timesnet",
    "fedformer",
    "itransformer_lstm",
    "lstm",
    "itransformer",
    "persistence",
]

RESULT_COLUMNS = [
    "site",
    "season",
    "model",
    "pred_len",
    "seed",
    "scale",
    "mae",
    "rmse",
    "r2",
    "mbe",
    "epochs_run",
    "train_seconds",
]


class Persistence(nn.Module):
    """Naive persistence: repeat the last observed target value over the horizon.

    NOTE: if the paper's Persistence [31] used a different rule (e.g. smart /
    clear-sky persistence or same-hour-yesterday), the numbers will differ.
    Calibrate against the paper at h=1 before trusting longer horizons.
    """

    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x):
        # x: [B, seq_len, enc_in]; channel 0 is the standardized target.
        return x[:, -1, 0:1].repeat(1, self.pred_len)


# Per-model architecture defaults. The round-1 configs for TimesNet / FEDformer
# / DLinear do not exist: round 1 never ran the Time-Series-Library in this
# codebase, so there is nothing to recover and nothing to calibrate against.
# Every model is therefore re-tuned under one protocol by run_optuna_baseline.py
# (validation-selected, same trial budget as GSTFM), and the values below are
# only the fallback used when no tuned config is on disk.
DEFAULT_ARCH = {
    "dlinear": dict(moving_avg=25),
    "timesnet": dict(d_model=64, d_ff=128, e_layers=2, top_k=5, num_kernels=6),
    "fedformer": dict(d_model=64, d_ff=128, e_layers=2, d_layers=1, n_heads=4,
                      moving_avg=25, modes=32, version="fourier", mode_select="random"),
    # round-1 reference/pre_iTransformer_LSTM_parallel.py:
    # params_dict={'hidden_dim':32,'layer_L':3,'layer_I':4,'heads':12,'dim_lstm':32}
    "itransformer_lstm": dict(dim_embed=32, depth=4, heads=12, dim_lstm=32, depth_lstm=3),
    "lstm": dict(hidden_size=128, num_layers=3),
    "itransformer": dict(dim=256, depth=5, heads=6),
    "persistence": {},
}


def tuned_arch_path(model, site, season, pred_len):
    return os.path.join(REPO_ROOT, "results",
                        f"best_params_{model}_{site}_{season}_h{pred_len}.json")


def load_tuned_arch(model, site, season, pred_len):
    """Load the validation-selected architecture for one cell, or None.

    Written by run_optuna_baseline.py. Keys starting with '_' are provenance
    metadata (study name, best value) and are not model arguments.
    """
    path = tuned_arch_path(model, site, season, pred_len)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        best = json.load(f)
    return {k: v for k, v in best.items() if not k.startswith("_")}


def build_model(name, seq_len, pred_len, enc_in=5, arch=None):
    """Instantiate a model by registry name. Output is always [B, pred_len].

    arch: dict of architecture overrides on top of DEFAULT_ARCH[name].
    """
    name = name.lower()
    if name not in DEFAULT_ARCH:
        raise ValueError(f"Unknown model '{name}'. Choices: {MODEL_NAMES}")
    cfg = dict(DEFAULT_ARCH[name])
    cfg.update(arch or {})

    if name in ("dlinear", "timesnet", "fedformer"):
        from models.baselines import build_baseline

        return build_baseline(name, seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, **cfg)
    if name == "itransformer_lstm":
        from models.iTransformer_LSTM import iTransformer_LSTM

        # forward(x[B,L,C]) -> [B, pred_len] (KAN head output sliced at [:,0,:])
        return iTransformer_LSTM(
            input_size=enc_in, length_pre=pred_len, length_input=seq_len, **cfg)
    if name == "lstm":
        from models.LSTM import LSTM

        # forward returns fc(last hidden) -> [B, output_size]
        return LSTM(input_size=enc_in, output_size=pred_len, **cfg)
    if name == "itransformer":
        from models.iTransformer import iTransformer_single

        # forward returns pred_list[0][:, :, 0] -> [B, pred_len]
        return iTransformer_single(
            num_variates=enc_in, lookback_len=seq_len, pred_length=pred_len,
            num_tokens_per_variate=1, use_reversible_instance_norm=True, **cfg)
    if name == "persistence":
        return Persistence(pred_len)
    raise ValueError(f"Unknown model '{name}'. Choices: {MODEL_NAMES}")


def resolve_csv_path(site, season):
    """Find data/<site>/<season>_<site>_*_h.csv (year span differs per site)."""
    pattern = os.path.join(REPO_ROOT, "data", site, f"{season}_{site}_*_h.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No data file matching {pattern}")
    return matches[0]


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def evaluate_dual(loader, model, criterion, device, scalar_y):
    """One test pass, metrics at both scales.

    Equivalent to utils.tools.evaluate with/without `scalar`, but robust for
    pred_len > 1 (applies scalar_y.mean_/scale_ directly instead of relying on
    StandardScaler.inverse_transform feature-count broadcasting).
    """
    model.eval()
    loss_sum = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.float().to(device)
            y_pre = model(x)
            loss_sum += criterion(y_pre, y).item() * x.size(0)
            preds.append(y_pre.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    test_loss = loss_sum / len(loader.dataset)
    std_metrics = metrics_of_pv(preds, trues)
    mean = float(scalar_y.mean_[0])
    scale = float(scalar_y.scale_[0])
    kw_metrics = metrics_of_pv(preds * scale + mean, trues * scale + mean)
    return test_loss, std_metrics, kw_metrics


def append_results(out_path, rows):
    """Append rows to the results CSV, writing the header if the file is new."""
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_done_keys(out_path):
    """Return the set of (site, season, model, pred_len, seed) already in the CSV."""
    done = set()
    out_path = os.path.abspath(out_path)
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add(
                    (
                        row["site"],
                        row["season"],
                        row["model"],
                        int(row["pred_len"]),
                        int(row["seed"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return done


def already_done(out_path, site, season, model, pred_len, seed):
    return (site, season, model, int(pred_len), int(seed)) in load_done_keys(out_path)


def _load_checkpoint(path):
    """Load a full pickled model (EarlyStopping saves via dill)."""
    import dill

    with open(path, "rb") as f:
        try:
            return torch.load(f, pickle_module=dill, weights_only=False)
        except TypeError:  # older torch without the weights_only kwarg
            f.seek(0)
            return torch.load(f, pickle_module=dill)


def main(args):
    same_seeds(args.seed)
    device = resolve_device(args.device)

    # ---- data ----
    csv_path = resolve_csv_path(args.site, args.season)
    df_all = pd.read_csv(csv_path, header=0)
    (
        data_train,
        data_valid,
        data_test,
        _ts_train,
        _ts_valid,
        _ts_test,
        scalar_y,
    ) = split_data_cnn(df_all, 0.8, 0.1, args.seq_len)
    enc_in = data_train.shape[1]

    multi_steps = args.pred_len > 1  # y: [B, pred_len] if multi_steps else [B, 1]
    ds_kwargs = dict(
        lookback_length=args.seq_len,
        lookforward_length=args.pred_len,
        multi_steps=multi_steps,
    )
    train_loader = DataLoader(
        data_detime(data=data_train, **ds_kwargs), batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        data_detime(data=data_valid, **ds_kwargs), batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        data_detime(data=data_test, **ds_kwargs), batch_size=args.batch_size, shuffle=False
    )

    # ---- model / training ----
    arch = dict(getattr(args, "arch", None) or {})
    if not arch and getattr(args, "use_tuned", True):
        arch = load_tuned_arch(args.model, args.site, args.season, args.pred_len) or {}
    model = build_model(args.model, args.seq_len, args.pred_len,
                        enc_in=enc_in, arch=arch).to(device)
    if arch:
        print(f"[info] arch overrides: {arch}", flush=True)
    criterion = nn.L1Loss(reduction="sum").to(device)  # MAE, per paper Table 1

    epochs_run = 0
    train_seconds = 0.0
    if args.model != "persistence":
        optm = optim.Adam(model.parameters(), lr=args.lr)
        # Same LR schedule as the authors' reference training scripts.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optm, mode="min", factor=0.5, patience=5)
        save_dir = os.path.join(REPO_ROOT, args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        # Shared checkpoint naming contract (consumed by scripts/dump_predictions.py):
        # results/ckpt/{model}_{site}_{season}_h{pred_len}_s{seed}.pt
        # NOTE(review): EarlyStopping pickles the FULL model via dill (not a bare
        # state_dict); dump_predictions.py supports both formats.
        ckpt = os.path.join(
            save_dir,
            f"{args.model}_{args.site}_{args.season}_h{args.pred_len}_s{args.seed}.pt",
        )
        earlystopping = EarlyStopping(ckpt, patience=args.patience, delta=args.delta)

        t0 = time.time()
        for epoch in range(args.epochs):
            train_loss = train(data=train_loader, model=model, criterion=criterion, optm=optm, device=device)
            valid_loss, ms = evaluate(data=valid_loader, model=model, criterion=criterion, device=device)
            scheduler.step(valid_loss)
            earlystopping(valid_loss, model)
            epochs_run = epoch + 1
            print(
                f"{args.model}|{args.site}|{args.season}|h={args.pred_len}|seed={args.seed}|"
                f"epoch {epochs_run}/{args.epochs}|train {train_loss:.4f}|valid {valid_loss:.4f}|"
                f"MAE {ms[0]:.4f}",
                flush=True,
            )
            if earlystopping.early_stop:
                print("Early stopping", flush=True)
                break
        train_seconds = round(time.time() - t0, 2)
        model = _load_checkpoint(ckpt).to(device)  # best model on valid loss

    # ---- validation metrics (hyper-parameter selection; never the test split) ----
    _, valid_m = evaluate(data=valid_loader, model=model, criterion=criterion, device=device)

    # ---- dual-scale test evaluation ----
    test_loss, std_m, kw_m = evaluate_dual(test_loader, model, criterion, device, scalar_y)
    print(
        f"[TEST] {args.model}|{args.site}|{args.season}|h={args.pred_len}|seed={args.seed}|"
        f"loss {test_loss:.4f}|std MAE {std_m[0]:.4f} RMSE {std_m[1]:.4f} R2 {std_m[2]:.4f}|"
        f"kw MAE {kw_m[0]:.4f} RMSE {kw_m[1]:.4f}",
        flush=True,
    )

    base = dict(
        site=args.site,
        season=args.season,
        model=args.model,
        pred_len=args.pred_len,
        seed=args.seed,
        epochs_run=epochs_run,
        train_seconds=train_seconds,
    )
    rows = [
        dict(base, scale="std", mae=std_m[0], rmse=std_m[1], r2=std_m[2], mbe=std_m[3]),
        dict(base, scale="kw", mae=kw_m[0], rmse=kw_m[1], r2=kw_m[2], mbe=kw_m[3]),
    ]
    if args.out:
        append_results(args.out, rows)

    return dict(base,
                valid=dict(zip(["mae", "rmse", "r2", "mbe"], valid_m)),
                std=dict(zip(["mae", "rmse", "r2", "mbe"], std_m)),
                kw=dict(zip(["mae", "rmse", "r2", "mbe"], kw_m)), test_loss=test_loss)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generic baseline trainer (paper Table 1 protocol, dual-scale metrics)."
    )
    parser.add_argument("--model", type=str, default="dlinear", choices=MODEL_NAMES)
    parser.add_argument("--site", type=str, default="7-First-Solar", choices=["7-First-Solar", "1B"])
    parser.add_argument("--season", type=str, default="Spring",
                        choices=["Spring", "Summer", "Autumn", "Winter"])
    parser.add_argument("--pred_len", type=int, default=1, help="forecast horizon (hours)")
    parser.add_argument("--seq_len", type=int, default=24, help="lookback length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda:0', ...")
    parser.add_argument("--save_dir", type=str, default=os.path.join("results", "ckpt"),
                        help="checkpoint directory (relative to repo root; shared "
                             "contract with scripts/dump_predictions.py)")
    parser.add_argument("--out", type=str, default=os.path.join("results", "single_runs.csv"),
                        help="results CSV (append mode); '' disables writing")
    parser.add_argument("--use_tuned", dest="use_tuned", action="store_true", default=True,
                        help="load results/best_params_{model}_{site}_{season}_h{H}.json "
                             "when present (default)")
    parser.add_argument("--no-use-tuned", dest="use_tuned", action="store_false",
                        help="force DEFAULT_ARCH instead of loading a tuned config")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
