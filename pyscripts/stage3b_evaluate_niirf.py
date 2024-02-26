# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from models import networks
from models.iir_downstream import HRTF_IIR
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.sofa_dataset import SOFAMultiple
from utils.util import get_log_mag_torch, seed_everything


def compute_metrics(dataset, model, render, config, pickle_name=None):
    device = next(model.parameters()).device

    # Specify the frequency range for computing LSD
    fwidth = 1 / config.nfft * config.sr
    low_idx, high_idx = int(np.ceil(20 / fwidth)), int(np.ceil(20000 / fwidth))

    # Compute the error over the dataset
    running_ae, running_se = [], []
    for data in tqdm(dataset):
        coord, hrir_oracle, sidx = [torch.tensor(x).to(device) for x in data]
        mag_oracle = get_log_mag_torch(hrir_oracle, config.nfft)

        gain, fc, fb = model(coord[None, 0], coord[None, 1], sidx[None])
        mag_estimate = render.get_log_mag(
            gain=gain[0, ...],
            fc=fc[0, ...],
            fb=fb[0, ...],
            nfft=config.nfft,
        )

        error = torch.abs(mag_oracle - mag_estimate)[..., low_idx:high_idx]
        error = error.detach().cpu().numpy()

        running_ae.append(error)
        running_se.append(np.square(error))

    mae = np.mean(running_ae)
    rmse = np.sqrt(np.mean(running_se))

    # Save the result in a pickle file if the filename is specified
    if pickle_name is not None:
        with open(pickle_name, "wb") as f:
            errors = running_ae
            pickle.dump(errors, f)

    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--directions", type=str, default="all")
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=10**15)
    args = parser.parse_args()
    path = Path(args.config_path)
    config = OmegaConf.load(path.joinpath("config.yaml"))
    seed_everything(0)

    # Prepare a directory for saving log
    os.makedirs(path.joinpath("log"), exist_ok=True)
    log_name = path.joinpath("log").joinpath("result.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)

    # Load the model and prepare render
    fcs = config.preprocess.fcs
    fc_limits = []
    for n in range(1, len(fcs) - 1):
        fc_limits.append((fcs[n - 1], fcs[n + 1]))

    model = getattr(networks, config.model.name)(fc_limits=fc_limits, **config.model.config)
    model.load_state_dict(torch.load(path.joinpath("best_adaptation.ckpt"), map_location="cpu"))
    model = model.to(config.device)
    model.eval()

    render = HRTF_IIR(npeaks=config.preprocess.n_bins)

    # Evaluate the model
    ref_range = (args.eval_start, args.eval_end)
    dataset = SOFAMultiple(
        config.dataset.name,
        config.dataset.target_sidxs,
        mode="tt",
        ref_range=ref_range,
        return_hrir=True,
    )
    if args.save_dir == "":
        metrics = compute_metrics(
            dataset,
            model,
            render,
            config=config.loss,
        )
        print(f"RMSE on {args.directions} directions: {metrics[1]}")

    else:
        metrics = compute_metrics(
            dataset,
            model,
            render,
            config=config.loss,
            pickle_name=f"{args.save_dir}/{args.directions}.pickle",
        )
        logging.info(f"RMSE on {args.directions} directions: {metrics[1]}")


if __name__ == "__main__":
    main()
