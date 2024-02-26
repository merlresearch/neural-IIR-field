# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2023 Chin-Yun Yu
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

"""
This script is adapted from this project
    (https://gist.githubusercontent.com/yoyololicon/f63f601d62187562070a61377cec9bf8/raw/151851a12c64c0e7b)
"""

import numpy as np
from scipy import signal


def extract_iir_params(H, nfft=1024, sr=44100, tap_of_smoothing=11, step=4):
    w = np.ones(tap_of_smoothing) / tap_of_smoothing
    smoothed_H = signal.fftconvolve(H, w, "same")
    freqs = np.arange(smoothed_H.shape[0]) / nfft * sr

    up_peaks, _ = signal.find_peaks(smoothed_H, prominence=0.6, height=(-20, None), distance=round(300 / sr * nfft))
    down_peaks, _ = signal.find_peaks(-smoothed_H, prominence=1, height=(None, 40), distance=round(300 / sr * nfft))

    if up_peaks[-1] == nfft // 2 - tap_of_smoothing // 2:
        up_peaks = up_peaks[:-1]
    if len(down_peaks) != 0:
        if down_peaks[-1] == nfft // 2 - tap_of_smoothing // 2:
            down_peaks = down_peaks[:-1]

    up_gain = np.maximum(H[up_peaks], 1)
    down_gain = np.minimum(H[down_peaks], -1)

    peaks = np.concatenate((up_peaks, down_peaks))
    gains = np.concatenate((up_gain, down_gain))

    left_slope = np.take(smoothed_H, peaks[:, None] + np.arange(step))
    left_slope -= np.take(smoothed_H, peaks[:, None] - np.arange(1, step + 1))
    left_slope = left_slope.mean(1)

    right_slope = np.take(smoothed_H, peaks[:, None] + np.arange(step))
    right_slope -= np.take(smoothed_H, peaks[:, None] + np.arange(1, step + 1))
    right_slope = right_slope.mean(1)

    slope = 0.5 * (np.abs(left_slope) + np.abs(right_slope))
    fb = sr / slope * 0.002
    fb = np.clip(fb, 200, 3000)

    return freqs, freqs[peaks], gains, fb
