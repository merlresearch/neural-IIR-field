# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def mse_nandetect(gt, est):
    error = torch.abs(gt - est)
    loss = torch.mean(torch.nan_to_num(torch.square(error), nan=0.0))
    return loss
