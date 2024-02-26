# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from pyscripts.models.networks import (
    MultiSubjectMagNFPEFT,
    MultiSubjectNIIRFPEFT,
    SingleSubjectMgaNF,
    SingleSubjectNIIRF,
)


@pytest.mark.parametrize("fc_limits", [[[100, 300], [200, 400], [300, 500], [400, 600]]])
@pytest.mark.parametrize("hidden_features", [16])
@pytest.mark.parametrize("hidden_layers", [1, 4])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dropout", [0])
@pytest.mark.parametrize("activation", ["PReLU", "GELU"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_single_subject_niirf(
    fc_limits,
    hidden_features,
    hidden_layers,
    scale,
    dropout,
    activation,
    batch_size,
):
    model = SingleSubjectNIIRF(
        fc_limits=fc_limits,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        scale=scale,
        dropout=dropout,
        activation=activation,
    )
    model.train()

    coordinate = torch.rand(batch_size, 3)
    gain, fc, fb = model(coordinate[:, 0], coordinate[:, 1])
    assert list(gain.shape) == [batch_size, 2, len(fc_limits) + 2]
    assert list(fc.shape) == [batch_size, 2, len(fc_limits) + 2]
    assert list(fb.shape) == [batch_size, 2, len(fc_limits)]

    gain[0].abs().mean().backward()


@pytest.mark.parametrize("hidden_features", [16])
@pytest.mark.parametrize("hidden_layers", [1, 4])
@pytest.mark.parametrize("out_features", [18])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dropout", [0])
@pytest.mark.parametrize("activation", ["PReLU", "GELU"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_single_subject_magnf(
    hidden_features,
    hidden_layers,
    out_features,
    scale,
    dropout,
    activation,
    batch_size,
):
    model = SingleSubjectMgaNF(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        scale=scale,
        dropout=dropout,
        activation=activation,
    )
    model.train()

    coordinate = torch.rand(batch_size, 3)
    magnitude = model(coordinate[:, 0], coordinate[:, 1])
    assert list(magnitude.shape) == [batch_size, 2, out_features // 2]

    magnitude[0].abs().mean().backward()


@pytest.mark.parametrize("fc_limits", [[[100, 300], [200, 400], [300, 500], [400, 600]]])
@pytest.mark.parametrize("hidden_features", [16])
@pytest.mark.parametrize("hidden_layers", [1, 4])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dropout", [0])
@pytest.mark.parametrize("n_listeners", [1, 4])
@pytest.mark.parametrize("activation", ["PReLU", "GELU"])
@pytest.mark.parametrize("peft_type", ["none", "lora", "bitfit", "lorabitfit"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_multi_subject_niirf(
    fc_limits,
    hidden_features,
    hidden_layers,
    scale,
    dropout,
    n_listeners,
    activation,
    peft_type,
    batch_size,
):
    model = MultiSubjectNIIRFPEFT(
        fc_limits=fc_limits,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        scale=scale,
        dropout=dropout,
        n_listeners=n_listeners,
        activation=activation,
        peft_type=peft_type,
    )
    model.train()

    coordinate = torch.rand(batch_size, 3)
    sidxs = torch.randint(low=0, high=n_listeners, size=(batch_size,))
    gain, fc, fb = model(coordinate[:, 0], coordinate[:, 1], sidxs)
    assert list(gain.shape) == [batch_size, 2, len(fc_limits) + 2]
    assert list(fc.shape) == [batch_size, 2, len(fc_limits) + 2]
    assert list(fb.shape) == [batch_size, 2, len(fc_limits)]

    gain[0].abs().mean().backward()


@pytest.mark.parametrize("hidden_features", [16])
@pytest.mark.parametrize("hidden_layers", [1, 4])
@pytest.mark.parametrize("out_features", [18])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dropout", [0])
@pytest.mark.parametrize("n_listeners", [1, 4])
@pytest.mark.parametrize("activation", ["PReLU", "GELU"])
@pytest.mark.parametrize("peft_type", ["none", "lora", "bitfit", "lorabitfit"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_multi_subject_magnf(
    hidden_features,
    hidden_layers,
    out_features,
    scale,
    dropout,
    n_listeners,
    activation,
    peft_type,
    batch_size,
):
    model = MultiSubjectMagNFPEFT(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        scale=scale,
        dropout=dropout,
        n_listeners=n_listeners,
        activation=activation,
        peft_type=peft_type,
    )
    model.train()

    coordinate = torch.rand(batch_size, 3)
    sidxs = torch.randint(low=0, high=n_listeners, size=(batch_size,))
    magnitude = model(coordinate[:, 0], coordinate[:, 1], sidxs)
    assert list(magnitude.shape) == [batch_size, 2, out_features // 2]

    magnitude[0].abs().mean().backward()
