#!/usr/bin/env bash
# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

expdir=$1

mkdir -p "${expdir}/original"
for i in `seq 1 96`; do
    wget "https://sofacoustics.org/data/database/hutubs/pp${i}_HRIRs_measured.sofa" -P "${expdir}/original"
done
