# GSTFM photovoltaic forecasting

This repository contains the reproducibility package for the EPJ Photovoltaics manuscript **A Gated-Fusion Spectro-Temporal Framework for Photovoltaic Power Forecasting with Time-Frequency Consistency Regularization**. GSTFM combines an iTransformer target branch, an LSTM covariate branch, cross-attention, a classical Adaptive Gated Covariate Fusion (AGCF) block and a Haar-wavelet consistency loss.

The package is deliberately separate from the authors' working vault. It contains the model and baseline implementations, training and sweep drivers, selected configuration JSON files, aggregate result files used for the manuscript tables, and input-data checksums. It does not contain raw DKASC measurements, checkpoints, credentials, Optuna databases, logs, or the manuscript source.

## Reproduce the reported results

Use Python 3.9+ and a CUDA-enabled PyTorch installation for training. Install the dependencies listed in `requirements.txt`, obtain the eight processed CSV files from the DKA Solar Centre, place them under `data/7-First-Solar/` and `data/1B/`, and run:

```bash
python scripts/check_inputs.py
python verify_facts.py
python run_multi_horizon.py --device cuda:0
python run_seed_gstfm.py --device cuda:0 --pred_lens 1 16 32 64
python run_seed_baselines.py --device cuda:0 --model timesnet
python run_ablation.py --device cuda:0
python run_site1b.py --device cuda:0
```

The selected architectures are stored in `results/best_params_*.json`. Search drivers use validation MAE and the manuscript protocol (30 trials per baseline; 40 trials for GSTFM) at H=1; selected architectures are then held fixed across H=16, 32 and 64. Run `python run_optuna_baseline.py --device cuda:0` and `python run_optuna_gstfm.py --device cuda:0` to repeat the searches.

The training loader fits a `StandardScaler` on the training rows only, applies it to validation and test rows, clips negative target power to zero, and uses a 24-hour lookback. The test windows and daylight counts are recorded in `data/input_manifest.json`. Direct multi-output horizons overlap at their forecast origins; they are not independent samples.

To compute reference skill scores from a result CSV:

```bash
python scripts/skill_scores.py results/multi_horizon.csv --out skill_scores.csv
```

The score is `1 - model_error / persistence_error`; positive values indicate an improvement over naive persistence.

## Input data and licensing

Raw measurements are intentionally not redistributed. See `data/README.md` for the required schema, source, split convention and test-window metadata. `paper_results/` contains aggregate CSV files supporting the reported tables and does not contain individual raw observations.

Third-party notices and licenses are in `THIRD_PARTY_NOTICES.md`, `models/baselines/LICENSE-TSLIB`, and `licenses/`. The repository's own code is released under the MIT license.
