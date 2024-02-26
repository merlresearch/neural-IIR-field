# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from pyscripts.utils.extract_iir_params import extract_iir_params


@pytest.mark.parametrize("nfft", [512, 1024])
@pytest.mark.parametrize("sr", [44100, 48000])
@pytest.mark.parametrize("tap_of_smoothing", [3, 11])
@pytest.mark.parametrize("step", [2, 4])
def test_mse_nandetect_wo_noise(nfft, sr, tap_of_smoothing, step):
    magnitude = torch.zeros(nfft // 2 + 1)
    magnitude[nfft // 8 - 1] = 10
    magnitude[nfft // 8 - 0] = 100
    magnitude[nfft // 8 + 1] = 10
    magnitude[nfft * 3 // 8 - 1] = -10
    magnitude[nfft * 3 // 8 - 0] = -100
    magnitude[nfft * 3 // 8 + 1] = -10
    extract_iir_params(magnitude, nfft=nfft, sr=sr, tap_of_smoothing=tap_of_smoothing, step=step)
