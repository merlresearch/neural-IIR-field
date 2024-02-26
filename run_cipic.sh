#!/usr/bin/env bash
# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


# Specify the directory for saving the data, configurations, and artifacts
expdir="exp/cipic"
modelname="model_example"


# Specify arrays of the subjects and number of measurements used for training, developement, and testing
all_subjects=(3 8 9 10 11 12 15 17 18 19 20 21 27 28 33 40 44 48 50 51 58 59 60 \
              61 65 119 124 126 127 131 133 134 135 137 147 148 152 153 154 155 \
              156 158 162 163 165)
subjects=(3)
amounts=(10 20 30 50 70 100 150)
ndata_tr=150
ndata_dev=100
ndata_tt=1000


# Stage 1
# Download the dataset and convert sofa files to pickles
mkdir -p $expdir
bash scripts/download_cipic.sh $expdir
python pyscripts/stage1_preprocess.py $expdir


# Stage 2
# Split the dataset into the training, developement, and test sets
for sidx in "${all_subjects[@]}"; do
    python pyscripts/stage2_split_dataset.py $expdir $sidx $ndata_tr $ndata_dev $ndata_tt
done
# If you use the default frequency range, you can skip the next line
# python pyscripts/compute_peak_and_notch_frequencies.py $expdir 32


# Stage 3
# Train and evaluate the proposed NIIRF
modeldir="${expdir}/stage3/${modelname}"
mkdir -p $modeldir
cp exp_example/cipic/stage3/model_example/config.yaml $modeldir/config.yaml

for sidx in "${subjects[@]}"; do
    printf -v _sidx "%05d" $sidx
    sdir="${modeldir}/s${_sidx}"

    for amount in "${amounts[@]}"; do
        printf -v _amount "%05d" $amount
        mkdir -p "${sdir}/${_amount}"

        # Modify the default config to fit to each condition
        new_yaml="${sdir}/${_amount}/config.yaml"
        cp "${modeldir}/config.yaml" $new_yaml
        original_sidxs=$(grep -oP 'target_sidxs: \[\K[^]]+' ${modeldir}/config.yaml)
        original_range=$(grep -oP 'ref_range: \[\K[^]]+' ${modeldir}/config.yaml)
        sed -i "s/${original_sidxs}/${sidx}/g" "${new_yaml}"
        sed -i "s/${original_range}/0, ${amount}/g" "${new_yaml}"

        # Train the proposed NIIRF under each condition
        python pyscripts/stage3a_train_niirf.py "${sdir}/${_amount}" adaptation
        # Evaluate the proposed NIIRF on the unseen directions
        python pyscripts/stage3b_evaluate_niirf.py "${sdir}/${_amount}" \
            --save_dir "${sdir}/${_amount}" \
            --directions unseen \
            --eval_start 0 \
            --eval_end $ndata_tt
    done
done


# Stage 4
# Train and evaluate a neural field estimating the HRTF magnitude
modeldir="${expdir}/stage4/${modelname}"
mkdir -p $modeldir
cp exp_example/cipic/stage4/model_example/config.yaml $modeldir/config.yaml

for sidx in "${subjects[@]}"; do
    printf -v _sidx "%05d" $sidx
    sdir="${modeldir}/s${_sidx}"

    for amount in "${amounts[@]}"; do
        printf -v _amount "%05d" $amount
        mkdir -p "${sdir}/${_amount}"

        # Modify the default config to fit to each condition
        new_yaml="${sdir}/${_amount}/config.yaml"
        cp "${modeldir}/config.yaml" $new_yaml
        original_sidxs=$(grep -oP 'target_sidxs: \[\K[^]]+' ${modeldir}/config.yaml)
        original_range=$(grep -oP 'ref_range: \[\K[^]]+' ${modeldir}/config.yaml)
        sed -i "s/${original_sidxs}/${sidx}/g" "${new_yaml}"
        sed -i "s/${original_range}/0, ${amount}/g" "${new_yaml}"

        # Train the neural field under each condition
        python pyscripts/stage4a_train_magnf.py "${sdir}/${_amount}" adaptation
        # Evaluate the neural field on the unseen directions
        python pyscripts/stage4b_evaluate_magnf.py "${sdir}/${_amount}" \
            --save_dir "${sdir}/${_amount}" \
            --directions unseen \
            --eval_start 0 \
            --eval_end $ndata_tt
    done
done
