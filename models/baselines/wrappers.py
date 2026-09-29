"""Wrappers for vendored Time-Series-Library models.

Inputs have shape [batch, lookback, 5], with target power in channel 0.
Outputs have shape [batch, horizon]. No future covariates or calendar
embeddings are supplied. Selected JSON configurations override defaults."""

import types

import torch
import torch.nn as nn

BASELINE_NAMES = ('dlinear', 'timesnet', 'fedformer')


def make_configs(name, seq_len, pred_len, enc_in=5, **overrides):
    """Build the argparse-style configs namespace expected by TSLib models.

    ``overrides`` sets any field on the namespace (d_model, d_ff, e_layers,
    n_heads, top_k, num_kernels, moving_avg, modes, version, mode_select, ...).
    Unknown keys are set anyway: TSLib models read configs by attribute, so an
    unused key is harmless, while silently dropping a real one would not be.
    """
    name = name.lower()
    if name not in BASELINE_NAMES:
        raise ValueError(
            "Unknown baseline '%s'; expected one of %s" % (name, list(BASELINE_NAMES)))

    configs = types.SimpleNamespace(
        # task / shape
        task_name='long_term_forecast',
        seq_len=seq_len,
        label_len=max(seq_len // 2, 1),
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        # architecture
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        dropout=0.1,
        activation='gelu',
        # embedding
        embed='timeF',
        freq='h',
        # Autoformer/FEDformer/DLinear decomposition
        factor=3,
        moving_avg=25,
        # TimesNet
        top_k=5,
        num_kernels=6,
    )
    if name == 'fedformer':
        configs.version = 'fourier'
        configs.mode_select = 'random'
        configs.modes = 32

    for key, val in overrides.items():
        setattr(configs, key, val)
    return configs


class TSLibForecaster(nn.Module):
    """Adapts a vendored TSLib model to forward(x[B,L,C]) -> y[B,pred_len].

    Handles time marks (None) and FEDformer-style decoder inputs internally,
    then slices the target channel (index 0) from the multivariate output.
    """

    TARGET_CHANNEL = 0  # Active_Power is the FIRST column in our CSVs

    def __init__(self, model, configs):
        super(TSLibForecaster, self).__init__()
        self.model = model
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

    def forward(self, x):
        # x: [B, seq_len, enc_in], standardized, target at channel 0
        batch = x.size(0)
        # Decoder input: last label_len observed steps + zero placeholders
        # (DLinear/TimesNet ignore it; FEDformer's forecast path expects the
        # standard TSLib decoder tensor shape [B, label_len + pred_len, C]).
        zeros = torch.zeros(batch, self.pred_len, self.enc_in,
                            dtype=x.dtype, device=x.device)
        x_dec = torch.cat([x[:, -self.label_len:, :], zeros], dim=1)
        # No calendar features in our pipeline -> x_mark_* = None
        # (vendored DataEmbedding supports x_mark=None).
        out = self.model(x, None, x_dec, None)  # [B, pred_len, c_out]
        return out[:, -self.pred_len:, self.TARGET_CHANNEL]  # [B, pred_len]


def _build_vendored(name, configs):
    """Instantiate the vendored TSLib Model (lazy imports keep optional
    dependencies of one model from breaking the others)."""
    if name == 'dlinear':
        from .tslib.DLinear import Model as DLinearModel
        return DLinearModel(configs)
    if name == 'timesnet':
        from .tslib.TimesNet import Model as TimesNetModel
        return TimesNetModel(configs)
    if name == 'fedformer':
        # NOTE: importing FEDformer pulls in MultiWaveletCorrelation, which
        # requires sympy + scipy at import time (even for version='fourier').
        from .tslib.FEDformer import Model as FEDformerModel
        return FEDformerModel(configs,
                              version=configs.version,
                              mode_select=configs.mode_select,
                              modes=configs.modes)
    raise ValueError(
        "Unknown baseline '%s'; expected one of %s" % (name, list(BASELINE_NAMES)))


def build_baseline(name, seq_len, pred_len, enc_in=5, **overrides):
    """Factory: build a wrapped baseline forecaster.

    Args:
        name: one of BASELINE_NAMES ('dlinear', 'timesnet', 'fedformer').
        seq_len: encoder lookback length (paper protocol: 24).
        pred_len: forecast horizon.
        enc_in: number of input channels (target first; default 5).
        **overrides: architecture hyper-parameters (see make_configs).

    Returns:
        nn.Module with forward(x[B, seq_len, enc_in]) -> y[B, pred_len].
    """
    name = str(name).lower()
    configs = make_configs(name, seq_len, pred_len, enc_in, **overrides)
    model = _build_vendored(name, configs)
    return TSLibForecaster(model, configs)
