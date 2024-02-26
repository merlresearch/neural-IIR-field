# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2023 Chin-Yun Yu
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

"""
This script is adapted from this project
    (https://github.com/yoyololicon/hrtf-notebooks/blob/master/HRTF%20cascaded%20IIR%20filter.ipynb)
"""
import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F


class LFS:
    def __init__(self, fs=44100):
        self.fs = fs

    def get_v(self, gain):
        return 10 ** (gain / 20.0)

    def get_poly(self, gain, fc, *args):
        V_0 = torch.clamp(self.get_v(gain), max=1)
        H_0 = self.get_v(gain) - 1
        tmp = torch.tan(np.pi * fc / self.fs)
        a = (tmp - V_0) / (tmp + V_0)

        b = torch.cat([1 + 0.5 * H_0 * (1 + a), a + 0.5 * H_0 * (a + 1)], -1)
        a = torch.cat([torch.ones_like(a), a], -1)
        return a, b

    def get_log_mag(self, gain, fc, *args, nfft=1024, eps=1.0e-3):
        a, b = self.get_poly(gain, fc)

        B = fft.rfft(b, nfft).abs().add(eps).log10()
        A = fft.rfft(a, nfft).abs().add(eps).log10()
        return (B - A) * 20


class HFS:
    def __init__(self, fs=44100):
        self.fs = fs

    def get_v(self, gain):
        return 10 ** (gain / 20.0)

    def get_poly(self, gain, fc, *args):
        V_0 = torch.clamp(self.get_v(gain), max=1)
        H_0 = self.get_v(gain) - 1
        tmp = torch.tan(np.pi * fc / self.fs)
        a = (tmp * V_0 - 1) / (tmp * V_0 + 1)

        b = torch.cat([1 + 0.5 * H_0 * (1 - a), a + 0.5 * H_0 * (a - 1)], -1)
        a = torch.cat([torch.ones_like(a), a], -1)
        return a, b

    def get_log_mag(self, gain, fc, *args, nfft=1024, eps=1.0e-3):
        a, b = self.get_poly(gain, fc)
        B = fft.rfft(b, nfft).abs().add(eps).log10()
        A = fft.rfft(a, nfft).abs().add(eps).log10()
        return (B - A) * 20


class Peak:
    def __init__(self, fs=44100):
        self.fs = fs

    def get_v(self, gain):
        return 10 ** (gain * 0.05)

    def get_poly(self, gain, fc, fb, *args):
        v0 = torch.clamp(self.get_v(gain), max=1)
        h0 = self.get_v(gain) - 1
        tmp = torch.tan(fb / self.fs * np.pi)
        d = -torch.cos(2 * np.pi * fc / self.fs)
        a = (tmp - v0) / (tmp + v0)

        b = torch.cat([1 + 0.5 * h0 * (1 + a), d * (1 - a), -a - 0.5 * h0 * (1 + a)], -1)
        a = torch.cat([torch.ones_like(a), d * (1 - a), -a], -1)
        return a, b

    def get_log_mag(self, gain, fc, fb, *args, nfft=1024, eps=1.0e-3):
        a, b = self.get_poly(gain, fc, fb)

        B = fft.rfft(b, nfft).abs().add(eps).log10()
        A = fft.rfft(a, nfft).abs().add(eps).log10()
        return (B - A) * 20


class HRTF_IIR:
    def __init__(self, npeaks=6, fs=44100):

        self.lfs = LFS(fs=fs)
        self.hfs = HFS(fs=fs)
        self.peaks = [Peak(fs=fs) for _ in range(npeaks)]
        self.cascade = [self.lfs] + self.peaks + [self.hfs]

    def get_poly(self, gain, fc, fb):
        alist, blist = [], []
        fb = F.pad(fb, (1, 1))  # for LFS and HFS
        for i, layer in enumerate(self.cascade):
            fci, gaini, fbi = fc[..., i, None], gain[..., i, None], fb[..., i, None]
            a, b = layer.get_poly(gaini, fci, fbi)

            alist.append(a)
            blist.append(b)
        return alist, blist

    def get_log_mag(self, gain, fc, fb, nfft=1024, eps=1.0e-3):
        y = 0.0
        fb = F.pad(fb, (1, 1))  # for LFS and HFS
        for i, layer in enumerate(self.cascade):
            gaini, fci, fbi = gain[..., i, None], fc[..., i, None], fb[..., i, None]
            logmag = layer.get_log_mag(gaini, fci, fbi, nfft=nfft, eps=eps)
            y = y + logmag
        return y

    def get_log_mag_all(self, gain, fc, fb, nfft=1024):
        logmags = []
        fb = F.pad(fb, (1, 1))  # for LFS and HFS
        for i, layer in enumerate(self.cascade):
            fci, gaini, fb = fc[..., i, None], gain[..., i, None], fb[..., i, None]
            logmag = layer.get_log_mag(gaini, fci, fb, nfft=nfft)
            logmags.append(logmag)
        return logmags
