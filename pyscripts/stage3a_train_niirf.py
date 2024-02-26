# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import networks
from models.iir_downstream import HRTF_IIR
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss_functions import mse_nandetect
from utils.sofa_dataset import SOFAMultiple
from utils.util import get_log_mag_torch, seed_everything


def forward(data, model, render, config):
    coord, hrir_oracle, sidx = data
    mag_oracle = get_log_mag_torch(hrir_oracle, config.nfft)

    gain, fc, fb = model(coord[:, 0], coord[:, 1], sidx)
    mag_estimate = render.get_log_mag(gain, fc, fb, nfft=config.nfft)

    loss_mag = mse_nandetect(mag_oracle, mag_estimate)
    loss_value = config.mag_weight * loss_mag
    return loss_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("mode", type=str)
    args = parser.parse_args()
    path = pathlib.Path(args.config_path)
    config = OmegaConf.load(path.joinpath("config.yaml"))
    seed_everything(config.seed)

    # Prepare a directory for saving log
    os.makedirs(path.joinpath("log"), exist_ok=True)
    log_name = path.joinpath("log").joinpath("exp.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)

    # Specify the training configuration
    if args.mode == "pretrain":
        target_sidxs = config.dataset.sidxs
        ref_range = None
        max_epoch = config.num_epoch
        ckpt_name = "best_pretrain.ckpt"
        optimizer_config = config.learning.optimizer
        if hasattr(config.learning, "scheduler"):
            scheduler_config = config.learning.scheduler
        else:
            scheduler_config = None

    else:
        target_sidxs = config.dataset.target_sidxs
        ref_range = config.dataset.ref_range
        max_epoch = config.adaptation_epoch
        ckpt_name = "best_adaptation.ckpt"
        optimizer_config = config.learning.adaptation_optimizer
        if hasattr(config.learning, "adaptation_scheduler"):
            scheduler_config = config.learning.adaptation_scheduler
        else:
            scheduler_config = None

    # Prepare dataset and dataloader
    tr_dataset = SOFAMultiple(
        config.dataset.name,
        target_sidxs,
        ref_range=ref_range,
        mode="tr",
        return_hrir=True,
    )
    dev_dataset = SOFAMultiple(
        config.dataset.name,
        target_sidxs,
        mode="dev",
        return_hrir=True,
    )

    if args.mode == "pretrain":
        tr_batch_size = config.batch_size
        dev_batch_size = config.batch_size
    else:
        tr_batch_size = min([config.adaptation_batch_size, len(tr_dataset) // 2])
        dev_batch_size = config.adaptation_batch_size

    tr_data_loader = DataLoader(
        tr_dataset,
        batch_size=tr_batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=dev_batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )

    # Initialize a neural field model
    fcs = config.preprocess.fcs
    fc_limits = []
    for n in range(1, len(fcs) - 1):
        fc_limits.append((fcs[n - 1], fcs[n + 1]))

    model = getattr(networks, config.model.name)(fc_limits=fc_limits, **config.model.config)

    # The following part is for loading a pre-trained model in adaptation
    if os.path.exists(path.joinpath("best_pretrain.ckpt")):
        logging.info("load pretrained weight")

        model.load_state_dict(torch.load(path.joinpath("best_pretrain.ckpt"), map_location="cpu"))
        for name, param in model.named_parameters():
            if "lora" in name or "bit" in name:
                if "bias" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    logging.info(f"adapt {name}")
            else:
                param.requires_grad = False

    else:
        logging.info("train from scrtach")

    model = model.to(config.device)
    render = HRTF_IIR(npeaks=config.preprocess.n_bins)

    # Prepare an optimizer and a scheduler
    optimizer = getattr(optim, optimizer_config.name)(model.parameters(), **optimizer_config.config)

    if scheduler_config is None:
        logging.info("w/o schedular")
    else:
        scheduler = getattr(optim.lr_scheduler, scheduler_config.name)(optimizer, **scheduler_config.config)
        logging.info("w/ schedular")

    # Iterate the training loop
    tr_loss, dev_loss = [], []
    dev_loss_min = 1.0e15
    early_stop = 0

    for epoch in range(max_epoch):

        running_loss = []
        model.train()
        for data in tqdm(tr_data_loader):
            data = [x.to(config.device) for x in data]
            model.zero_grad()
            loss_value = forward(data, model, render, config.loss)
            loss_value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.learning.clip)
            optimizer.step()
            running_loss.append(loss_value.item())

        tr_loss.append(np.mean(running_loss))

        running_loss = []
        model.eval()
        for data in tqdm(dev_data_loader):
            data = [x.to(config.device) for x in data]
            loss_value = forward(data, model, render, config.loss)
            running_loss.append(loss_value.item())

        dev_loss.append(np.mean(running_loss))

        if scheduler_config is not None:
            scheduler.step(dev_loss[-1])

        logging.info(f"Epoch {epoch}")
        logging.info(f"tr_loss {tr_loss[-1]}, dev_loss {dev_loss[-1]}")

        if dev_loss[-1] <= dev_loss_min:
            dev_loss_min = dev_loss[-1]
            early_stop = 0
            torch.save(model.state_dict(), path.joinpath(ckpt_name))
        else:
            early_stop += 1

        if early_stop == config.patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        if np.isnan(dev_loss[-1]):
            logging.info("Loss is Nan. Training should be stopped")
            break


if __name__ == "__main__":
    main()
