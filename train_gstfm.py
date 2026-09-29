"""Train GSTFM with classical AGCF and one-level Haar regularization.

Selected architecture JSON files are read by the sweep drivers. Metrics are
written in standardized units and kW; model selection uses validation MAE."""

import argparse
import csv
import json
import os
import sys
import time
import warnings

# Repo root = directory containing this file; make local packages importable.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import dill
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import split_data_cnn, data_detime
from models.iTransformer_LSTM_agcf import iTransformer_LSTM
from utils.losses import WaveletConsistencyLoss
from utils.tools import EarlyStopping, same_seeds, train, evaluate

warnings.filterwarnings('ignore')

RESULTS_CSV_HEADER = ['site', 'season', 'model', 'pred_len', 'seed', 'scale',
                      'mae', 'rmse', 'r2', 'mbe', 'epochs_run', 'train_seconds']
MODEL_NAME = 'GSTFM'


def str2bool(v):
    """Argparse-friendly boolean parser (accepts true/false/1/0/yes/no)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')


def build_arg_parser():
    parser = argparse.ArgumentParser(description='GSTFM (proposed model) trainer, Optuna-free')
    # data
    parser.add_argument('--site', type=str, default='7-First-Solar',
                        choices=['7-First-Solar', '1B'])
    parser.add_argument('--season', type=str, default='Spring',
                        choices=['Spring', 'Summer', 'Autumn', 'Winter'])
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--lookback', type=int, default=24)
    # training protocol (paper Table 1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    # model knobs (classical AGCF; legacy CLI names)
    parser.add_argument('--gate_type', type=str, default=None,
                        choices=['agcf', 'se', 'none'],
                        help="gate generator; overrides --use_agcf when given. "
                             "'agcf' = proposed AGCF, 'se' = squeeze-excitation, 'none' = no gate")
    parser.add_argument('--head', type=str, default='kan', choices=['kan', 'mlp'],
                        help='readout head; mlp is the ablation replacement for KAN')
    parser.add_argument('--variant', type=str, default='full',
                        help="ablation tag; written to the CSV 'model' column as "
                             "GSTFM-<variant> (except 'full' -> GSTFM) so that "
                             "variants never collide in the resume key")
    parser.add_argument('--use_agcf', type=str2bool, default=True)
    parser.add_argument('--gate_residual', type=str2bool, default=True)
    parser.add_argument('--gate_channel', type=str2bool, default=False)
    parser.add_argument('--gate_width', type=int, default=6)
    parser.add_argument('--gate_depth', type=int, default=3)
    # model knobs (backbone)
    parser.add_argument('--dim_embed', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--dim_lstm', type=int, default=128)
    parser.add_argument('--depth_lstm', type=int, default=3)
    # wavelet-consistency loss knobs
    parser.add_argument('--lambda_A', type=float, default=0.05)
    parser.add_argument('--lambda_D', type=float, default=0.05)
    parser.add_argument('--wavelet_levels', type=int, default=1)
    # misc
    parser.add_argument('--device', type=str, default=None,
                        help="e.g. 'cuda:0' or 'cpu'; default auto-detect")
    parser.add_argument('--results_csv', type=str,
                        default=os.path.join(REPO_ROOT, 'results', 'gstfm_runs.csv'))
    parser.add_argument('--force', action='store_true',
                        help='re-run even if this combo already exists in the results CSV')
    return parser


def parse_args():
    return build_arg_parser().parse_args()


# Backbone hyper-parameters an Optuna best-trial JSON may override. Anything
# else in the JSON is ignored, so a stale/extra key cannot silently change the
# training protocol.
#
# 'use_agcf' / 'gate_type' are NOT here on purpose: whether the gate exists is
# the paper's claim, not something a search may switch off. A tuned JSON can
# shape the gate (gate_width, gate_depth, gate_channel) but never remove it, so
# the model named GSTFM is the same architecture in every table.
TUNABLE_KEYS = ('dim_embed', 'depth', 'heads', 'dim_lstm', 'depth_lstm',
                'gate_channel', 'gate_residual', 'gate_width', 'gate_depth')


def best_params_path(site, season, pred_len):
    return os.path.join(REPO_ROOT, 'results',
                        f'best_params_{site}_{season}_h{pred_len}.json')


def apply_best_params(args, path=None, required=False):
    """Overlay an Optuna best-trial JSON onto args, in place.

    Returns the path used, or None when no file was found (and required=False).
    """
    path = path or best_params_path(args.site, args.season, args.pred_len)
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(
                f'no tuned config at {path}; run run_optuna_gstfm.py first '
                f'(or pass an explicit --best_params)')
        print(f'[warn] no tuned config at {path}; using CLI/default hyper-parameters')
        return None
    with open(path, 'r', encoding='utf-8') as f:
        best = json.load(f)
    applied = {k: best[k] for k in TUNABLE_KEYS if k in best}
    for k, v in applied.items():
        setattr(args, k, v)
    print(f'[info] loaded tuned config from {path}: {applied}')
    return path


def data_file_path(site, season):
    """Derive the CSV path: years span differs per site."""
    years = '2019_2022' if site == '7-First-Solar' else '2019_2023'
    return os.path.join(REPO_ROOT, 'data', site, f'{season}_{site}_{years}_h.csv')


def variant_model_name(variant):
    """CSV 'model' label for a run: 'full' is the headline model, others are tagged."""
    variant = (variant or 'full').strip()
    return MODEL_NAME if variant == 'full' else f'{MODEL_NAME}-{variant}'


def existing_scales(results_csv, site, season, pred_len, seed, model_name=MODEL_NAME):
    """Return the set of scales already recorded for this run combo."""
    scales = set()
    if not os.path.isfile(results_csv):
        return scales
    with open(results_csv, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if (row.get('site') == site and row.get('season') == season
                    and row.get('model') == model_name
                    and str(row.get('pred_len')) == str(pred_len)
                    and str(row.get('seed')) == str(seed)):
                scales.add(row.get('scale'))
    return scales


def append_result_row(results_csv, row):
    """Append one row, writing the header first if the file is new/empty."""
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    need_header = (not os.path.isfile(results_csv)) or os.path.getsize(results_csv) == 0
    with open(results_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(RESULTS_CSV_HEADER)
        writer.writerow(row)


def main(args=None):
    args = args if args is not None else parse_args()
    model_name = variant_model_name(args.variant)

    # Resume-friendly skip: both scales already recorded -> nothing to do.
    done = existing_scales(args.results_csv, args.site, args.season, args.pred_len, args.seed,
                           model_name)
    if {'std', 'kw'} <= done and not args.force:
        print(f'[skip] {model_name} {args.site}/{args.season} pred_len={args.pred_len} '
              f'seed={args.seed} already in {args.results_csv} (use --force to re-run)')
        return

    same_seeds(args.seed)
    device = torch.device(args.device if args.device is not None
                          else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(f'[info] device={device}')

    # ---------------- data ----------------
    file_path = data_file_path(args.site, args.season)
    df_all = pd.read_csv(file_path, header=0)
    multi_steps = args.pred_len > 1
    (data_train, data_valid, data_test,
     ts_train, ts_valid, ts_test, scalar_y) = split_data_cnn(df_all, 0.8, 0.1, args.lookback)
    dataset_train = data_detime(data=data_train, lookback_length=args.lookback,
                                lookforward_length=args.pred_len, multi_steps=multi_steps)
    dataset_valid = data_detime(data=data_valid, lookback_length=args.lookback,
                                lookforward_length=args.pred_len, multi_steps=multi_steps)
    dataset_test = data_detime(data=data_test, lookback_length=args.lookback,
                               lookforward_length=args.pred_len, multi_steps=multi_steps)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # ---------------- model / loss / optimizer ----------------
    model = iTransformer_LSTM(input_size=5, length_pre=args.pred_len,
                              length_input=args.lookback,
                              dim_lstm=args.dim_lstm, depth_lstm=args.depth_lstm,
                              dim_embed=args.dim_embed, depth=args.depth, heads=args.heads,
                              use_agcf=args.use_agcf, gate_type=args.gate_type,
                              head=args.head,
                              gate_width=args.gate_width,
                              gate_depth=args.gate_depth,
                              gate_residual=args.gate_residual,
                              gate_channel=args.gate_channel).to(device)
    print(f'[info] gate_type={model.gate_type} head={model.head_type} '
          f'lambda_A={args.lambda_A} lambda_D={args.lambda_D}')

    criterion = WaveletConsistencyLoss(lambda_A=args.lambda_A, lambda_D=args.lambda_D,
                                       levels=args.wavelet_levels,
                                       base='mae', reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # ---------------- checkpoint paths ----------------
    ckpt_dir = os.path.join(REPO_ROOT, 'results', 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = (f'{model_name}_{args.site}_{args.season}_h{args.pred_len}_s{args.seed}'
           .replace('/', '_'))
    ckpt_path = os.path.join(ckpt_dir, f'{tag}.pt')
    sidecar_path = os.path.join(ckpt_dir, f'{tag}.json')
    # EarlyStopping (utils.tools) pickles the FULL model with dill to this temp file.
    tmp_full_path = os.path.join(ckpt_dir, f'{tag}_full_tmp.pt')
    earlystopping = EarlyStopping(tmp_full_path, patience=10, delta=0.0001)

    # ---------------- training loop ----------------
    epochs_run = 0
    train_start = time.time()
    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train(data=train_loader, model=model, criterion=criterion,
                               optm=optimizer, device=device)
            valid_loss, ms = evaluate(data=valid_loader, model=model, criterion=criterion,
                                      device=device)
            scheduler.step(valid_loss)
            earlystopping(valid_loss, model)
            epochs_run = epoch + 1
            print(f'\n{model_name}|{args.site}/{args.season}|epoch {epochs_run}/{args.epochs}'
                  f'|time:{time.time() - t0:.2f}s'
                  f'|lr:{optimizer.state_dict()["param_groups"][0]["lr"]:.6f}\n'
                  f'Loss_train:{train_loss:.4f}|Loss_valid:{valid_loss:.4f}'
                  f'|MAE:{ms[0]:.4f}|RMSE:{ms[1]:.4f}|R2:{ms[2]:.4f}|MBE:{ms[3]:.4f}',
                  flush=True)
            if earlystopping.early_stop:
                print('Early stopping')
                break
    except KeyboardInterrupt:
        # Do NOT fall through to the result rows: a half-trained model would be
        # written to the CSV as a finished run, and every later resume would
        # skip that cell because both scales are present.
        print('Training interrupted by user; nothing written to the results CSV.')
        raise
    train_seconds = round(time.time() - train_start, 2)

    # ---------------- restore best checkpoint ----------------
    if os.path.isfile(tmp_full_path):
        try:
            model = torch.load(tmp_full_path, pickle_module=dill,
                               map_location=device, weights_only=False)
        except TypeError:  # older torch without the weights_only kwarg
            model = torch.load(tmp_full_path, pickle_module=dill, map_location=device)
        model = model.to(device)
    else:
        print('[warn] no early-stopping checkpoint found; using last-epoch weights')

    # Persist a clean state_dict checkpoint + json sidecar of args.
    torch.save(model.state_dict(), ckpt_path)
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if os.path.isfile(tmp_full_path):
        os.remove(tmp_full_path)
    print(f'[info] checkpoint saved to {ckpt_path}')

    # ---------------- validation metrics (model selection / Optuna objective) ----------
    # Reported on the standardized scale, from the early-stopped best checkpoint.
    # Hyper-parameter search must key off this, never off the test split.
    _, ms_valid = evaluate(data=valid_loader, model=model, criterion=criterion, device=device)

    # ---------------- dual-scale test evaluation (verification item V1) ----------------
    # std scale: metrics on standardized values (paper-reported values / 100)
    _, ms_std = evaluate(data=test_loader, model=model, criterion=criterion, device=device)
    # kw scale: metrics after inverse-transforming with the target scaler
    _, ms_kw = evaluate(data=test_loader, model=model, criterion=criterion, device=device,
                        scalar=scalar_y)
    print(f'[test/std] MAE:{ms_std[0]:.4f}|RMSE:{ms_std[1]:.4f}|R2:{ms_std[2]:.4f}|MBE:{ms_std[3]:.4f}')
    print(f'[test/kw ] MAE:{ms_kw[0]:.4f}|RMSE:{ms_kw[1]:.4f}|R2:{ms_kw[2]:.4f}|MBE:{ms_kw[3]:.4f}')

    for scale, ms in (('std', ms_std), ('kw', ms_kw)):
        append_result_row(args.results_csv,
                          [args.site, args.season, model_name, args.pred_len, args.seed,
                           scale, ms[0], ms[1], ms[2], ms[3], epochs_run, train_seconds])
    print(f'[info] results appended to {args.results_csv}')

    return {'model_name': model_name,
            'valid_mae': ms_valid[0], 'valid_rmse': ms_valid[1],
            'test_mae_std': ms_std[0], 'test_rmse_std': ms_std[1],
            'test_mae_kw': ms_kw[0], 'test_rmse_kw': ms_kw[1],
            'epochs_run': epochs_run, 'train_seconds': train_seconds}


if __name__ == '__main__':
    main()
