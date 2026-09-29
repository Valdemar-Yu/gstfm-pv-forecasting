"""Baseline forecasters (DLinear / TimesNet / FEDformer) vendored from
thuml/Time-Series-Library (MIT, see LICENSE-TSLIB) with a uniform factory.

Usage:
    from models.baselines import build_baseline, BASELINE_NAMES
    model = build_baseline('dlinear', seq_len=24, pred_len=1, enc_in=5)
    y_hat = model(x)  # x: [B, 24, 5] -> y_hat: [B, 1]
"""

from .wrappers import build_baseline, BASELINE_NAMES, make_configs, TSLibForecaster

__all__ = ['build_baseline', 'BASELINE_NAMES', 'make_configs', 'TSLibForecaster']
