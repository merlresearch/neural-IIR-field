# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def classify_text(input_text):
    mapping = {"tr": ["tr", "train"], "tt": ["tt", "test", "eval"], "dev": ["dev", "developement", "val", "valid"]}

    for label, keywords in mapping.items():
        if any(keyword in input_text.lower() for keyword in keywords):
            return label

    return "Unknown"


class SOFAMultiple(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        sidxs=[],
        ref_range=None,
        mode="tr",
        return_hrir=False,
    ):

        self.coord = []
        self.sidxs = []
        self.hrirs = []
        self.low_hrirs = []
        self.return_hrir = return_hrir

        mode = classify_text(mode)
        self.pkls = []
        path = pathlib.Path(dataset_path).joinpath("stage2")
        if ref_range is None:
            for sidx in sidxs:
                temp = path.joinpath(f"s{sidx:05}")
                pnmaes = list(temp.joinpath(mode).glob("*.pickle"))
                self.pkls += pnmaes
                self.sidxs += [sidx - 1] * len(pnmaes)
        else:
            for sidx in sidxs:
                temp = path.joinpath(f"s{sidx:05}")
                meta = OmegaConf.load(temp.joinpath("meta.yaml"))
                didxs = getattr(meta, f"{mode}_didxs")
                for n in range(*ref_range):
                    fname = f"s{sidx:05}_d{didxs[n]:05}.pickle"
                    pname = temp.joinpath(mode).joinpath(fname)
                    self.pkls.append(pname)
                    self.sidxs.append(sidx - 1)

        for pkl in tqdm(self.pkls):
            with open(pkl, "rb") as f:
                self.coord.append(pickle.load(f) / 360 * 2 * np.pi)
                hrir = pickle.load(f)
                self.hrirs.append(hrir)

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        return self.coord[idx], self.hrirs[idx], self.sidxs[idx]
