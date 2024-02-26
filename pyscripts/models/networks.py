# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LFS_FC_LIMITS = (20, 200)
HFS_FC_LIMITS = (16000, 22000)
FB_LIMITS = (100, 4000)


class MLP(nn.Module):
    def __init__(self, input_features, output_features, *args, dropout=0.0, bias=True, activation="GELU", **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_features, output_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activatin = getattr(nn, activation)()

    def forward(self, x):
        x = self.activatin(self.fc(x))
        x = self.dropout(x)
        return x


class LoRAMLP(nn.Module):
    def __init__(self, input_features, output_features, *args, dropout=0.0, bias=True, activation="GELU", **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_features, output_features, bias=bias)
        self.activatin = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, u=0, v=0, bias=0.0):
        z = u * torch.mean(v * x, -1, keepdim=True)
        x = self.fc(x) + z + bias
        x = self.dropout(self.activatin(x))
        return x


class SingleSubjectNIIRF(nn.Module):
    def __init__(
        self,
        fc_limits,
        hidden_features=128,
        hidden_layers=1,
        scale=1,
        dropout=0.1,
        activation="GELU",
    ):
        super().__init__()
        assert hidden_features % 2 == 0
        self.out_features = (3 * len(fc_limits) + 4) * 2
        self.n_peaks = len(fc_limits)

        fc_low = [LFS_FC_LIMITS[0]] + [x[0] for x in fc_limits] + [HFS_FC_LIMITS[0]]
        self.fc_low = torch.nn.Parameter(
            torch.tensor(fc_low, dtype=torch.float32),
            requires_grad=False,
        )
        fc_width = [LFS_FC_LIMITS[1] - LFS_FC_LIMITS[0]]
        fc_width += [x[1] - x[0] for x in fc_limits]
        fc_width += [HFS_FC_LIMITS[1] - HFS_FC_LIMITS[0]]
        self.fc_width = torch.nn.Parameter(
            torch.tensor(fc_width, dtype=torch.float32),
            requires_grad=False,
        )

        self.fb_low = FB_LIMITS[0]
        self.fb_width = FB_LIMITS[1] - FB_LIMITS[0]

        # For random Fourier feature
        bmat = scale * np.random.default_rng(0).normal(0.0, 1.0, (hidden_features // 2, 2))
        self.bmat = torch.nn.Parameter(torch.tensor(bmat.astype(np.float32)), requires_grad=False)

        # For MLP
        mlps = []
        for _ in range(hidden_layers):
            mlps.append(MLP(hidden_features, hidden_features, dropout=dropout, activation=activation))
        self.mlps = nn.Sequential(*mlps)
        self.out_linear = nn.Linear(hidden_features, self.out_features)

    def forward(self, phis, thetas, *args, **kwargs):
        """
        Input:
            phis: (batch_size, )
            thetas: (batch_size, )
        """
        bs = phis.shape[0]
        emb = torch.stack([phis - np.pi, thetas], -1) @ self.bmat.T
        emb = torch.concatenate([emb.sin(), emb.cos()], axis=-1)
        emb = self.mlps(emb)
        estimate = self.out_linear(emb)

        estimate = estimate.reshape(bs, 2, -1)
        gain, fc, fb = torch.split(estimate, self.n_peaks + 2, dim=-1)
        fc = self.fc_width[None, None, :] * torch.sigmoid(fc) + self.fc_low[None, None, :]
        fb = self.fb_width * torch.sigmoid(fb) + self.fb_low
        return gain, fc, fb


class SingleSubjectMgaNF(nn.Module):
    def __init__(
        self,
        hidden_features=128,
        hidden_layers=1,
        out_features=1026,
        scale=1,
        dropout=0.1,
        activation="GELU",
    ):
        super().__init__()
        assert hidden_features % 2 == 0
        assert out_features % 2 == 0

        # For random Fourier feature
        bmat = scale * np.random.default_rng(0).normal(0.0, 1.0, (hidden_features // 2, 2))
        self.bmat = torch.nn.Parameter(torch.tensor(bmat.astype(np.float32)), requires_grad=False)

        # For MLP
        mlps = []
        for _ in range(hidden_layers):
            mlps.append(MLP(hidden_features, hidden_features, dropout=dropout, activation=activation))
        self.mlps = nn.Sequential(*mlps)
        self.out_linear = nn.Linear(hidden_features, out_features)

    def forward(self, phis, thetas, *args, **kwargs):
        """
        Input:
            phis: (batch_size, )
            thetas: (batch_size, )
        """
        bs = phis.shape[0]
        emb = torch.stack([phis, thetas], -1) @ self.bmat.T
        emb = torch.concatenate([emb.sin(), emb.cos()], axis=-1)
        emb = self.mlps(emb)
        estimate = self.out_linear(emb)

        estimate = estimate.reshape(bs, 2, -1)
        return estimate


class MultiSubjectNIIRFPEFT(nn.Module):
    def __init__(
        self,
        fc_limits,
        hidden_features=128,
        hidden_layers=1,
        scale=1,
        dropout=0.1,
        n_listeners=1000,
        activation="GELU",
        peft_type="lora",
        **kwargs
    ):
        super().__init__()

        assert hidden_features % 2 == 0
        self.out_features = (3 * len(fc_limits) + 4) * 2
        self.n_peaks = len(fc_limits)
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        fc_low = [LFS_FC_LIMITS[0]] + [x[0] for x in fc_limits] + [HFS_FC_LIMITS[0]]
        self.fc_low = torch.nn.Parameter(torch.tensor(fc_low, dtype=torch.float32), requires_grad=False)
        fc_width = [LFS_FC_LIMITS[1] - LFS_FC_LIMITS[0]]
        fc_width += [x[1] - x[0] for x in fc_limits]
        fc_width += [HFS_FC_LIMITS[1] - HFS_FC_LIMITS[0]]
        self.fc_width = torch.nn.Parameter(torch.tensor(fc_width, dtype=torch.float32), requires_grad=False)
        self.fb_low = FB_LIMITS[0]
        self.fb_width = FB_LIMITS[1] - FB_LIMITS[0]

        # For random Fourier feature
        bmat = scale * np.random.default_rng(0).normal(0.0, 1.0, (hidden_features // 2, 2))
        self.bmat = torch.nn.Parameter(torch.tensor(bmat.astype(np.float32)), requires_grad=False)

        # For MLP
        mlps = []
        bias = "bit" not in peft_type
        for _ in range(hidden_layers):
            mlps.append(LoRAMLP(hidden_features, hidden_features, dropout=dropout, bias=bias, activation=activation))
        self.mlps = nn.Sequential(*mlps)
        self.out_linear = LoRAMLP(hidden_features, self.out_features, dropout=0.0, bias=bias, activation="Identity")

        # For listener embedding
        self.n_listeners = n_listeners
        self.use_bitfit = "bit" in peft_type
        self.use_lora = "lora" in peft_type

        if "bit" in peft_type:
            bitfit = [nn.Linear(n_listeners, hidden_features) for _ in range(hidden_layers)]
            bitfit.append(nn.Linear(n_listeners, self.out_features))
            self.bitfit = nn.ModuleList(bitfit)

        if "lora" in peft_type:
            lora_a = [nn.Linear(n_listeners, hidden_features) for _ in range(hidden_layers)]
            lora_a.append(nn.Linear(n_listeners, self.out_features))
            for layer in lora_a:
                nn.init.zeros_(layer.weight)
            self.lora_a = nn.ModuleList(lora_a)

            lora_b = [nn.Linear(n_listeners, hidden_features) for _ in range(hidden_layers)]
            lora_b.append(nn.Linear(n_listeners, hidden_features))
            self.lora_b = nn.ModuleList(lora_b)

    def forward(self, phis, thetas, sidxs, *args, **kwargs):
        """
        Input:
            phis: (batch_size, )
            thetas: (batch_size, )
            sidx: (batch_size,)
        """
        bs = phis.shape[0]
        onehot = F.one_hot(sidxs, self.n_listeners).type(torch.float32)

        emb = torch.stack([phis - np.pi, thetas], -1) @ self.bmat.T
        x = torch.concatenate([emb.sin(), emb.cos()], axis=-1)
        for n in range(self.hidden_layers):
            x = self.mlps[n](
                x,
                u=self.lora_a[n](onehot) if self.use_lora else 0,
                v=self.lora_b[n](onehot) if self.use_lora else 0,
                bias=self.bitfit[n](onehot) if self.use_bitfit else 0,
            )

        estimate = self.out_linear(
            x,
            u=self.lora_a[-1](onehot) if self.use_lora else 0,
            v=self.lora_b[-1](onehot) if self.use_lora else 0,
            bias=self.bitfit[-1](onehot) if self.use_bitfit else 0,
        )

        estimate = estimate.reshape(bs, 2, -1)
        gain, fc, fb = torch.split(estimate, self.n_peaks + 2, dim=-1)
        fc = self.fc_width[None, None, :] * torch.sigmoid(fc) + self.fc_low[None, None, :]
        fb = self.fb_width * torch.sigmoid(fb) + self.fb_low
        return gain, fc, fb


class MultiSubjectMagNFPEFT(nn.Module):
    def __init__(
        self,
        hidden_features=128,
        hidden_layers=1,
        out_features=514,
        scale=1,
        dropout=0.1,
        n_listeners=1000,
        activation="GELU",
        peft_type="lorabitfit",
        **kwargs
    ):
        super().__init__()
        assert hidden_features % 2 == 0
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.out_features = out_features

        # For random Fourier feature
        bmat = scale * np.random.default_rng(0).normal(0.0, 1.0, (hidden_features // 2, 2))
        self.bmat = torch.nn.Parameter(torch.tensor(bmat.astype(np.float32)), requires_grad=False)

        # For MLP
        mlps = []
        bias = "bit" not in peft_type
        for _ in range(hidden_layers):
            mlps.append(LoRAMLP(hidden_features, hidden_features, dropout=dropout, bias=bias, activation=activation))
        self.mlps = nn.Sequential(*mlps)

        self.out_linear = LoRAMLP(hidden_features, self.out_features, dropout=0.0, bias=bias, activation="Identity")
        # For listener embedding
        self.n_listeners = n_listeners
        self.bitfit = "bit" in peft_type
        self.lora = "lora" in peft_type

        if "bit" in peft_type:
            bitfit = [nn.Linear(n_listeners, hidden_features) for _ in range(hidden_layers)]
            bitfit.append(nn.Linear(n_listeners, self.out_features))
            self.bitfit = nn.ModuleList(bitfit)

        if "lora" in peft_type:
            lora_a = [nn.Linear(n_listeners, hidden_features) for _ in range(hidden_layers)]
            lora_a.append(nn.Linear(n_listeners, self.out_features))
            for layer in lora_a:
                nn.init.zeros_(layer.weight)
            self.lora_a = nn.ModuleList(lora_a)

            lora_b = [nn.Linear(n_listeners, hidden_features) for _ in range(hidden_layers)]
            lora_b.append(nn.Linear(n_listeners, hidden_features))
            self.lora_b = nn.ModuleList(lora_b)

    def forward(self, phis, thetas, sidxs):
        """
        Input:
            phis: (batch_size, )
            thetas: (batch_size, )
            sidx: (batch_size,)
        """
        bs = phis.shape[0]
        onehot = F.one_hot(sidxs, self.n_listeners).type(torch.float32)

        emb = torch.stack([phis - np.pi, thetas], -1) @ self.bmat.T
        x = torch.concatenate([emb.sin(), emb.cos()], axis=-1)

        for n in range(self.hidden_layers):
            x = self.mlps[n](
                x,
                u=self.lora_a[n](onehot) if self.lora else 0,
                v=self.lora_b[n](onehot) if self.lora else 0,
                bias=self.bitfit[n](onehot) if self.bitfit else 0,
            )

        estimate = self.out_linear(
            x,
            u=self.lora_a[-1](onehot) if self.lora else 0,
            v=self.lora_b[-1](onehot) if self.lora else 0,
            bias=self.bitfit[-1](onehot) if self.bitfit else 0,
        )

        estimate = estimate.reshape(bs, 2, -1)
        return estimate
