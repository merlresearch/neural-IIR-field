# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from pyscripts.utils.loss_functions import mse_nandetect


@pytest.mark.parametrize("batch_size", [[1], [4, 1], [4, 4], [4, 4, 4]])
def test_mse_nandetect_wo_noise(batch_size):
    ground_truth = torch.randn(*batch_size)
    mse_val = mse_nandetect(ground_truth, ground_truth)
    torch.testing.assert_close(mse_val, torch.tensor(0.0), rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("batch_size", [[1], [4, 1], [4, 4], [4, 4, 4]])
def test_mse_nandetect_w_noise(batch_size):
    ground_truth = torch.randn(*batch_size)
    noise = torch.randn(*batch_size)
    estimate = ground_truth + noise
    mse_val = mse_nandetect(ground_truth, estimate)
    torch_mse = torch.nn.MSELoss()(ground_truth, estimate)
    torch.testing.assert_close(mse_val, torch_mse, rtol=1.0e-5, atol=1.0e-6)
