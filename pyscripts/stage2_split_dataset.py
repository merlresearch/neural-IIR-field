# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf
from utils.util import get_didx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("subidx", type=int)
    parser.add_argument("tr_amount", type=int)
    parser.add_argument("dev_amount", type=int)
    parser.add_argument("tt_amount", type=int)
    args = parser.parse_args()
    meta = OmegaConf.load(Path(args.dataset_path).joinpath("meta.yaml"))

    assert hasattr(meta.subjects, f"s{args.subidx:05}")
    subject = getattr(meta.subjects, f"s{args.subidx:05}")
    ndirections = subject.ndirections
    temp = args.tr_amount + args.dev_amount + args.tt_amount
    assert temp == ndirections, (temp, ndirections)

    path = Path(subject.path)
    pkls_rest = sorted(list(path.glob("*.pickle")))
    rng = np.random.default_rng(0)
    pkls_rest = rng.permutation(pkls_rest)

    newpath = Path(*["stage2" if x == "stage1" else x for x in path.parts])
    newmeta = {
        "tr_amount": args.tr_amount,
        "dev_amount": args.dev_amount,
        "tt_amount": args.tt_amount,
    }

    for mode in ["tr", "dev", "tt"]:
        pkls_mode = pkls_rest[: getattr(args, f"{mode}_amount")]
        pkls_rest = pkls_rest[getattr(args, f"{mode}_amount") :]

        save_path = newpath.joinpath(mode)
        os.makedirs(save_path, exist_ok=True)
        newmeta[f"{mode}_didxs"] = []

        for src in pkls_mode:
            dst = save_path.joinpath(src.name)
            os.symlink(Path(os.getcwd()).joinpath(src), dst)
            newmeta[f"{mode}_didxs"].append(get_didx(src.name))

    assert len(pkls_rest) == 0

    fname = newpath.joinpath("meta.yaml")

    with open(fname, "w") as f:
        yaml.dump(newmeta, f, default_flow_style=False)


if __name__ == "__main__":
    main()
