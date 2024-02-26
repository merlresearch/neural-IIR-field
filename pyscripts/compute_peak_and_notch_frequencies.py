# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import pickle

import numpy as np
from tqdm import tqdm
from utils.extract_iir_params import extract_iir_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", type=str)
    parser.add_argument("npeaks", type=int, default=32)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--fc_lowend", type=int, default=100)
    parser.add_argument("--fc_highend", type=int, default=22000)
    args = parser.parse_args()
    exp_path = pathlib.Path(args.exp_path)

    assert "cipic" in args.exp_path or "hutubs" in args.exp_path
    pickles = []
    if "cipic" in args.exp_path:
        sidxs = [
            3,
            8,
            9,
            10,
            11,
            12,
            15,
            17,
            18,
            19,
            20,
            21,
            27,
            28,
            33,
            40,
            44,
            48,
            50,
            51,
            58,
            59,
            60,
            61,
            65,
            119,
            124,
            126,
            127,
            131,
            133,
            134,
            135,
            137,
            147,
            148,
            152,
            153,
            154,
            155,
        ]
        sr = 44100
    else:
        sidxs = range(1, 88)
        sr = 44100

    for sidx in sidxs:
        sidx_path = exp_path.joinpath("stage2").joinpath(f"s{sidx:05}")
        pickles += sidx_path.joinpath("tr").glob("*.pickle")

    peaks = []
    for pkl in tqdm(pickles):
        with open(pkl, "rb") as f:
            _ = pickle.load(f)
            hrir = pickle.load(f)

        for ch in range(2):
            mag = np.abs(np.fft.rfft(hrir[ch, :], args.nfft))
            mag = 20 * np.log10(mag + np.max(mag) * 1e-3)
            _, peak_notch_freq, _, _ = extract_iir_params(mag, nfft=args.nfft, sr=sr)
            peaks += list(peak_notch_freq)

    peaks.sort()
    fc = [args.fc_lowend]
    for n in range(0, args.npeaks):
        tmp = peaks[int((n + 0.5) * len(peaks) / args.npeaks)]
        fc.append(int(np.round(tmp)))

    fc.append(args.fc_highend)
    print("Frequencies for designing the frquency range of fc: ", *fc)


if __name__ == "__main__":
    main()
