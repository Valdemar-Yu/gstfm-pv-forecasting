import math
from typing import Optional

import torch
import torch.nn as nn
from models import iTransformer_block, CrossAttention
from models.KAN import KAN

# Classical AGCF implementation. The public package contains the classical bounded MLP gate used in the reported runs.

class BoundedMLPGate(nn.Module):
    """GELU MLP with a tanh-bounded output.
    Returns values in [-1, 1] and is fully differentiable.
    """
    def __init__(self, in_dim: int, gate_width: int, depth: int, out_dim: int):
        super().__init__()
        hidden = max(16, 2 * gate_width)
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(enc)) 


class ClassicalAGCF(nn.Module):
    """Classical AGCF generator producing per-head gates g in (0,1).
    Inputs:  x_pool [B, in_dim], optional t_emb (concatenated if given)
    Output:  g      [B, H]

    in_dim must be given at construction: a lazily created proj_in would be
    absent from model.parameters() when the optimizer is built, so its weights
    would never be updated.
    """
    def __init__(self, heads: int, gate_width: int = 6, depth: int = 3, in_dim: int = None):
        super().__init__()
        self.heads = heads
        self.gate_width = gate_width
        self.depth = depth

        exp_out = gate_width
        self.gate_mlp = BoundedMLPGate(in_dim=gate_width, gate_width=gate_width,
                                  depth=depth, out_dim=exp_out)

        # Bounded projection: input -> latent gate descriptor
        if in_dim is None:
            raise ValueError('ClassicalAGCF requires in_dim at construction')
        self.proj_in = nn.Linear(in_dim, gate_width, bias=True)
        # linear map from bounded gate features to heads, then sigmoid to (0,1)
        self.proj = nn.Linear(exp_out, heads)

    def forward(self, x_pool: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = x_pool if t_emb is None else torch.cat([x_pool, t_emb], dim=-1)
        z_bounded = torch.tanh(self.proj_in(z)) * math.pi  # bounded numeric range

        gate_features = self.gate_mlp(z_bounded)

        g = torch.sigmoid(self.proj(gate_features))  # [B, H]
        return g


class SEGatingUnit(nn.Module):
    """Standard squeeze-and-excitation gate generator.

    Drop-in replacement for ClassicalAGCF: same signature, same output
    shape [B, heads] in (0, 1). Only the gate *generator* differs, so an
    AGCF-vs-SE comparison isolates the gating mechanism and nothing else.
    """

    def __init__(self, heads: int, in_dim: int, reduction: int = 4):
        super().__init__()
        self.heads = heads
        hidden = max(heads, in_dim // reduction)
        self.squeeze = nn.Linear(in_dim, hidden, bias=True)
        self.excite = nn.Linear(hidden, heads, bias=True)

    def forward(self, x_pool: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = x_pool if t_emb is None else torch.cat([x_pool, t_emb], dim=-1)
        return torch.sigmoid(self.excite(torch.relu(self.squeeze(z))))  # [B, H]


class iTransformer_LSTM(nn.Module):
    """iTransformer+LSTM with a pluggable gate around the fusion path.

    Args (new knobs)
    ---------
    gate_type:      'agcf' (classical AGCF) | 'se'
                    (standard squeeze-excitation) | 'none' (no gate)
    use_agcf:        deprecated bool alias. True -> gate_type='agcf',
                    False -> gate_type='none'. Ignored when gate_type is given
                    explicitly.
    head:           'kan' (proposed readout) | 'mlp' (conventional 2-layer MLP)
    gate_width:       latent descriptor width (default 6)
    gate_depth:      GELU hidden layers in the MLP (default 3)
    gate_residual:  apply residual gating on CrossAttention output (default True)
    gate_channel:   apply channel gating before the readout head (default False)
    """

    def __init__(self, input_size=5,  length_pre=1, dim_lstm=128, depth_lstm=3,
                 length_input=48, dim_embed=128, depth=4, heads=6,
                 use_agcf: bool = False, gate_width: int = 6, gate_depth: int = 3,
                 gate_residual: bool = True, gate_channel: bool = False,
                 gate_type: Optional[str] = None, head: str = 'kan'):
        super(iTransformer_LSTM, self).__init__()
        if gate_type is None:
            gate_type = 'agcf' if use_agcf else 'none'
        gate_type = str(gate_type).lower()
        if gate_type not in ('agcf', 'se', 'none'):
            raise ValueError("gate_type must be 'agcf', 'se' or 'none', got %r" % gate_type)
        head = str(head).lower()
        if head not in ('kan', 'mlp'):
            raise ValueError("head must be 'kan' or 'mlp', got %r" % head)

        self.length_pre = length_pre
        self.dim_embed = dim_embed
        self.heads = heads
        self.gate_type = gate_type
        self.head_type = head
        self.use_agcf = (gate_type != 'none')  # kept: downstream code branches on it
        self.gate_residual = gate_residual
        self.gate_channel = gate_channel

        # iTransformer branch for target variable
        self.model1 = iTransformer_block(num_variates=1, lookback_len=length_input,
                                         pred_length=length_pre, dim=dim_embed, depth=depth, heads=heads,
                                         num_tokens_per_variate=1, use_reversible_instance_norm=True)

        # LSTM branch for covariates (input_size-1)
        self.lstm = nn.LSTM(input_size=input_size-1,
                            hidden_size=dim_lstm,
                            num_layers=depth_lstm,
                            batch_first=True,
                            bidirectional=False)

        # Cross-modal fusion
        self.cross = CrossAttention(dim=dim_embed, lenth=dim_lstm)

        # Gate generator: classical AGCF, SE, or none
        # gate input is x1_base.mean(dim=1) -> [B, dim_embed]
        if gate_type == 'agcf':
            self.agcf = ClassicalAGCF(heads=heads, gate_width=gate_width, depth=gate_depth,
                                         in_dim=dim_embed)
        elif gate_type == 'se':
            self.agcf = SEGatingUnit(heads=heads, in_dim=dim_embed)
        else:
            self.agcf = None
        # map per-head gates to scalar residual gate and per-channel gate
        if self.agcf is not None:
            self.to_res_gate = nn.Linear(heads, 1)
            self.to_ch_gate = nn.Linear(heads, dim_embed)

        # Readout head: [*, dim_embed] -> [*, length_pre]
        if head == 'kan':
            self.k_mpl = KAN([dim_embed, length_pre])
        else:
            self.k_mpl = nn.Sequential(
                nn.Linear(dim_embed, dim_embed),
                nn.SiLU(),
                nn.Linear(dim_embed, length_pre),
            )

    def forward(self, x):
        # x: [B, T, C] with first channel the target, others covariates
        x2, _ = self.lstm(x[:, :, 1:])                  # [B, T, dim_lstm]
        x1_base = self.model1(x[:, :, 0, None])         # [B, Lp, D]
        x1_cross = self.cross(x1_base, x2)              # [B, Lp, D]

        if self.use_agcf:
            # pooled context from the target stream; no extra data needed
            x_pool = x1_base.mean(dim=1)                # [B, D]
            g_heads = self.agcf(x_pool, None)            # [B, H]

            # residual gate: scalar per sample in (0,1)
            if self.gate_residual:
                g_res = torch.sigmoid(self.to_res_gate(g_heads)).unsqueeze(1)  # [B,1,1]
                x1 = x1_base + g_res * x1_cross
            else:
                x1 = x1_cross

            # channel gate before KAN: per-channel scaling
            if self.gate_channel:
                g_ch = torch.sigmoid(self.to_ch_gate(g_heads)).unsqueeze(1)    # [B,1,D]
                x1 = x1 * (1.0 + 0.25 * g_ch)  # gentle modulation
        else:
            x1 = x1_cross

        # KAN head expects [B, Lp, D] -> [B, Lp, length_pre]
        output = self.k_mpl(x1)                           # [B, Lp, length_pre]
        return output[:, 0, :]                            # keep original return shape [B, length_pre]
