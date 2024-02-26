# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random
import re

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_didx(fname):
    match = re.search(r"_d(\d+)", str(fname))
    return int(match.group(1))


def get_subidx(subject):
    match = re.search(r"\d+", subject)
    return int(match.group())


def get_log_mag_torch(x, nfft, eps=1.0e-3):
    y = torch.log10(torch.abs(torch.fft.rfft(x, nfft)) + eps)
    return 20.0 * y
