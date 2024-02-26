#!/usr/bin/env bash
# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


# Specify the directory for saving the data, configurations, and artifacts
expdir="exp/hutubs"
modelname="model_example"


# Specify arrays of the subjects and number of measurements used for adaptation
subjects=(89)
amounts=(10 20 30 50 100)
all_directions=440
seen_directions=100
unseen_directions=100
ndata_adaptation_tr=150


# Stage 1
# Download the dataset and convert sofa files to pickles
mkdir -p $expdir
bash scripts/download_hutubs.sh $expdir
python pyscripts/stage1_preprocess.py $expdir


# Stage 2
# Create the training and developement sets for pre-training, and
# split the mesuraments into the training, developement test sets for adaptation
# The following two group of subjects will be used in the pre-training of NIIRF
ndata_tr=$(($all_directions - $unseen_directions))
ndata_dev=0
ndata_tt=$unseen_directions
for sidx in `seq 1 77`; do
    python pyscripts/stage2_split_dataset.py $expdir $sidx $ndata_tr $ndata_dev $ndata_tt
done
ndata_tr=$(($all_directions - $seen_directions - $unseen_directions))
ndata_dev=$seen_directions
ndata_tt=$unseen_directions
for sidx in `seq 78 87`; do
    python pyscripts/stage2_split_dataset.py $expdir $sidx $ndata_tr $ndata_dev $ndata_tt
done
# The following group of subjects will be used for evaluating the adaptation capability of NIIRF
ndata_tr=$ndata_adaptation_tr
ndata_dev=$(($all_directions - $ndata_tr - $seen_directions - $unseen_directions))
ndata_tt=$(($seen_directions + $unseen_directions))
for sidx in `seq 89 95`; do
    python pyscripts/stage2_split_dataset.py $expdir $sidx $ndata_tr $ndata_dev $ndata_tt
done
# If you use the default frequency range, you can skip the next line
# python pyscripts/compute_peak_and_notch_frequencies.py $expdir 32


# Stage 3
# Pre-train, adapt, and evaluate the proposed NIIRF
modeldir="${expdir}/stage3/${modelname}"
mkdir -p $modeldir
cp exp_example/hutubs/stage3/model_example/config.yaml $modeldir/config.yaml
# Pre-train NIIRF on multiple subjects
python pyscripts/stage3a_train_niirf.py $modeldir pretrain

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

        # Adapt the pre-trained NIIRF to each of new subjects with different amount of measurements
        ln -fs "../../best_pretrain.ckpt" "${sdir}/${_amount}/best_pretrain.ckpt"
        python pyscripts/stage3a_train_niirf.py "${sdir}/${_amount}" adaptation
        # Evaluate the adapted NIIRF on the seen and unseen directions
        python pyscripts/stage3b_evaluate_niirf.py "${sdir}/${_amount}" \
            --save_dir "${sdir}/${_amount}" \
            --directions seen \
            --eval_start 0 \
            --eval_end $seen_directions
        python pyscripts/stage3b_evaluate_niirf.py "${sdir}/${_amount}" \
            --save_dir "${sdir}/${_amount}" \
            --directions unseen \
            --eval_start $seen_directions \
            --eval_end $(($seen_directions + $unseen_directions))
    done
done


# Stage 4
# Pre-train, adapt, and evaluate a neural field estimating the HRTF magnitude
modeldir="${expdir}/stage4/${modelname}"
mkdir -p $modeldir
cp exp_example/hutubs/stage4/model_example/config.yaml $modeldir/config.yaml
# Pre-train the neural field on multiple subjects
python pyscripts/stage4a_train_magnf.py $modeldir pretrain
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

        # Adapt the pre-trained neural field to each of new subjects with different amount of measurements
        ln -fs "../../best_pretrain.ckpt" "${sdir}/${_amount}/best_pretrain.ckpt"
        python pyscripts/stage4a_train_magnf.py "${sdir}/${_amount}" adaptation
        # Evaluate the adapted neural field on the seen and unseen directions
        python pyscripts/stage4b_evaluate_magnf.py "${sdir}/${_amount}" \
            --save_dir "${sdir}/${_amount}" \
            --directions seen \
            --eval_start 0 \
            --eval_end $seen_directions
        python pyscripts/stage4b_evaluate_magnf.py "${sdir}/${_amount}" \
            --save_dir "${sdir}/${_amount}" \
            --directions unseen \
            --eval_start $seen_directions \
            --eval_end $(($seen_directions + $unseen_directions))
    done
done
