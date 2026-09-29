"""Wavelet-consistency training losses for GSTFM.

dwt_haar() and WaveletConsistencyLoss are extracted VERBATIM from
reference/Quasa.py (the authors' original gated+wavelet training script).

WARNING
-------
WaveletConsistencyLoss reduces to the plain base loss (e.g. MAE) whenever
``y.shape[1] < 2`` -- i.e. for single-step prediction (pred_len == 1) the
wavelet consistency terms are silently skipped because a length-1 sequence
cannot be decomposed by the Haar DWT. This mirrors the original
implementation exactly; do NOT change this behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 小波变换
def dwt_haar(x, levels=1):
    """
    x: [B, T]  (T>=2)
    return: (A, D)  低频A与高频D按最后一级返回；如需全部级别可改为列表累积
    """
    # Haar 滤波器
    h = torch.tensor([1/2**0.5, 1/2**0.5], device=x.device, dtype=x.dtype).view(1,1,2)
    g = torch.tensor([1/2**0.5,-1/2**0.5], device=x.device, dtype=x.dtype).view(1,1,2)

    def _dwt1(z):
        # z:[B,T] -> A,D:[B,T/2]
        if z.shape[1] % 2 == 1:            # 奇数长度做尾部复制填充
            z = F.pad(z.unsqueeze(1), (0,1), mode='replicate').squeeze(1)
        zc = z.unsqueeze(1)                 # [B,1,T]
        A = F.conv1d(zc, h, stride=2).squeeze(1)
        D = F.conv1d(zc, g, stride=2).squeeze(1)
        return A, D

    A, D = _dwt1(x)
    for _ in range(levels-1):
        A, _D = _dwt1(A)                    # 只在A上继续分解
        D = _D                               # 按需也可返回每层的D
    return A, D

class WaveletConsistencyLoss(nn.Module):
    def __init__(self, lambda_A=0.1, lambda_D=0.1, levels=1,
                 base='mae', reduction='sum'):
        super().__init__()
        self.la, self.ld = lambda_A, lambda_D
        self.levels = levels
        self.base = base
        self.reduction = reduction

    def _base_loss(self, y_hat, y):
        if self.base == 'mae':
            return F.l1_loss(y_hat, y, reduction=self.reduction)
        elif self.base == 'mse':
            return F.mse_loss(y_hat, y, reduction=self.reduction)
        elif self.base == 'huber':
            return F.smooth_l1_loss(y_hat, y, reduction=self.reduction)
        else:
            return F.l1_loss(y_hat, y, reduction=self.reduction)

    def forward(self, y_hat, y):
        if y_hat.dim() == 3 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        if y.dim() == 3 and y.size(-1) == 1:
            y = y.squeeze(-1)

        base = self._base_loss(y_hat, y)
        if y.shape[1] < 2:
            return base

        A_hat, D_hat = dwt_haar(y_hat, levels=self.levels)
        A,     D     = dwt_haar(y,     levels=self.levels)

        loss_A = F.l1_loss(A_hat, A, reduction=self.reduction)
        loss_D = F.l1_loss(D_hat, D, reduction=self.reduction)
        return base + self.la * loss_A + self.ld * loss_D
