# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import sofa
import yaml
from tqdm import tqdm
from utils.util import get_subidx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    args = parser.parse_args()
    path = Path(args.dataset_path)
    subjects = list(path.joinpath("original").glob("*.sofa"))

    meta = {"nsubjects": len(subjects), "subjects": {}}
    for subject in tqdm(sorted(subjects)):
        subidx = get_subidx(str(subject))
        spath = path.joinpath("stage1").joinpath(f"s{subidx:05}")
        os.makedirs(spath, exist_ok=True)

        sofa_file = sofa.Database.open(subject)
        sr = sofa_file.Data.SamplingRate.get_values()
        hrir = sofa_file.Data.IR.get_values().astype(np.float32)
        sphcoord = sofa_file.SourcePosition.get_values().astype(np.float32)
        ndirections = hrir.shape[0]

        for didx in range(ndirections):
            fname = spath.joinpath(f"s{subidx:05}_d{didx:05}.pickle")
            with open(fname, mode="wb") as f:
                pickle.dump(sphcoord[didx, ...], f)
                pickle.dump(hrir[didx, ...], f)

        smeta = {"path": str(spath), "sr": int(sr), "ndirections": ndirections}
        meta["subjects"][f"s{subidx:05}"] = smeta

    fname = path.joinpath("meta.yaml")

    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False)


if __name__ == "__main__":
    main()
