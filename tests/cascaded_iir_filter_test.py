# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from pyscripts.models.iir_downstream import HRTF_IIR


@pytest.mark.parametrize("npeaks", [8, 32])
@pytest.mark.parametrize("fs", [44100, 48000])
@pytest.mark.parametrize("nfft", [512, 1024])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_hrtf_iir_filter_forawrd(npeaks, fs, nfft, batch_size):
    gain = torch.randn(batch_size, 2, npeaks + 2)
    fc = fs / 2 * torch.rand(batch_size, 2, npeaks + 2)
    fb = 1000 * torch.rand(batch_size, 2, npeaks + 2)

    render = HRTF_IIR(npeaks=npeaks, fs=fs)
    estimate = render.get_log_mag(
        gain=gain,
        fc=fc,
        fb=fb,
        nfft=nfft,
    )

    assert list(estimate.shape) == [batch_size, 2, nfft // 2 + 1]

    alist, blist = render.get_poly(
        gain=gain,
        fc=fc,
        fb=fb,
    )

    assert len(alist) == len(blist)
    assert len(alist) == npeaks + 2

    for idx, (a, b) in enumerate(zip(alist, blist)):
        if idx == 0 or idx == npeaks + 1:
            assert list(a.shape) == [batch_size, 2, 2]
            assert list(b.shape) == [batch_size, 2, 2]
        else:
            assert list(a.shape) == [batch_size, 2, 3]
            assert list(b.shape) == [batch_size, 2, 3]
