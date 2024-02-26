#!/usr/bin/env bash
# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

expdir=$1

mkdir -p "${expir}/original"
subjects=(3 8 9 10 11 12 15 17 18 19 20 21 27 28 33 40 44 48 50 51 58 59 60 61 65 119 124 126 127 131 133 134 135 137 147 148 152 153 154 155 156 158 162 163 165)

for i in "${subjects[@]}"; do
    printf -v j "%03d" $i
    wget "https://sofacoustics.org/data/database/cipic/subject_${j}.sofa" -P "${expdir}/original"
done
