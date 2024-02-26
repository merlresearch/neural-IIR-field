<!--
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# NIIRF: Neural IIR Filter Field for HRTF Upsampling and Personalization

This repository includes source code for training and evaluating the neural infinite impulse response filter field (NIIRF) proposed in the following ICASSP 2024 paper:

    @InProceedings{Masuyama2024ICASSP_niirf,
      author    =  {Masuyama, Yoshiki and Wichern, Gordon and Germain, Fran\c{c}ois G. and Pan, Zexu and Khurana, Sameer and Hori, Chiori and {Le Roux}, Jonathan},
      title     =  {NIIRF: Neural IIR Filter Field for HRTF Upsampling and Personalization},
      booktitle =  {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year      =  2024,
      month     =  apr
    }

## Table of contents

1. [Environment setup](#environment-setup)
2. [Training and evaluating single-subject neural fields](#training-and-evaluating-single-subject-neural-fields)
3. [Training and evaluating multi-subject neural fields](#training-and-evaluating-multi-subject-neural-fields)
4. [Contributing](#contributing)
5. [Copyright and license](#copyright-and-license)

## Environment setup

The code has been tested using `python 3.10.9` on Linux.
Necessary dependencies can be installed using the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

***

## Training and evaluating single-subject neural fields

In order to train and evaluate neural fields on the CIPIC dataset, please execute `run_cipic.sh` after specifying `EXPDIR`.
The dataset, model checkpoints, and training/evaluation log files will be stored in it.
`run_cipic.sh` consists of the following four stages:

- **Stage 1:**
    - This stage downloads the dataset and converts the sofa files to pickles.
    - The sofa files of the 44 released subjects will be downloaded from [the sofa conventions](https://sofacoustics.org/data/database/cipic/).
    - The pickles will be stored in `EXPDIR/stage1/s{subject_id}`.

- **Stage 2:**
    - This stage splits the training, developement, and test sets by `python pyscripts/stage2_split_dataset.py EXPDIR SIDX NDATA_TR NDATA_DEV NDATA_TT`
    - A sum of `NDATA_TR`, `NDATA_DEV`, and `NDATA_TT` must be the number of HRTF measurements for each subject, 1250.
    - The created training, dev, and test sets respectively saved in `EXPDIR/stage2/s{subject_id}`.

- **Stage 3:**
    - This stage trains the proposed neural field, NIIRF, for each subject with different amounts of measurements and evaluates its interpolation capability.
    - As a default, `config.yaml` in [exp_example](exp_example/cipic/stage3/model_example/) will be copied to your `MODELDIR` and used for the training.
    - Log-spectral distortion (LSD) on the entire test set will be reported in `EXPDIR/stage3/MODELNAME/s{subject_id}/{amount_of_measurements}/log`.

- **Stage 4:**
    - This stage trains a neural field that directly estimates magnitude HRTF for each subject with different amounts of measurements and evaluates it.
    - As a default, `config.yaml` in [exp_example](exp_example/cipic/stage4/model_example/) will be copied to your `MODELDIR` and used for the training.
    - Log-spectral distortion (LSD) on the entire test set will be reported in `EXPDIR/stage4/MODELNAME/s{subject_id}/{amount_of_measurements}/log`.

The following table shows LSD for different numbers of measurements.
This result is only for the subject3, while the paper reported LSD averaged over all the subjects.
| method       |  10 |  20 |  30 |  50 |  70 | 100 | 150 |
|-------------:|----:|----:|----:|----:|----:|----:|----:|
| Mag. NF      | 6.6 | 5.5 | 4.6 | 3.9 | 3.6 | 3.3 | 3.1 |
| NIIRF (K=32) | 6.1 | 5.0 | 4.4 | 3.9 | 3.5 | 3.4 | 3.1 |


## Training and evaluating multi-subject neural fields

In order to train and evaluate our method on the HUTUBS dataset, please execute `run_hutubs.sh` after specifying `EXPDIR`.
`run_hutubs.sh` consists of the four stages similar to `run_cipic.sh`, but this experiment pre-trains a multi-subject neural field and adapts it to another subject in Stage 3 and Stage 4.
For the adaptation method, the current implementation supports only LoRA and BitFit.

- **Stage 3:**
    - NIIRF is pre-trained with HRTFs of multiple subjects by `python pyscripts/stage3a_train_niirf.py MODELDIR pretrain`.
    - The pre-trained model is adapted to each subject with different amounts of measurements by `python pyscripts/stage3a_train_niirf.py SDIR/_AMOUNT adaptation`.
    - The adapted model will be evaluated on both seen and unseen directions.
        - Seen directions (100 directions in default): HRTFs of non-target subjects for these directions are used in the pre-training.
        - Unseen directions (100 directions in default): HRTFs for these directions are used in neither pre-training nor adaptation.

- **Stage 4:**
    - Similar to stage 3, a magnitude neural field is pre-trained with multiple subjects and adapted to the target subject.


The following table shows LSD for different numbers of measurements, where LoRA is used for the adaptation.
This result is only for subject 89, while the paper reported LSD averaged over subjects ID from 89 to 95.
| method       | directions |  10 |  20 |  30 |  50 | 100 |
|-------------:|-----------:|----:|----:|----:|----:|----:|
| Mag. NF      | seen       | 4.0 | 3.8 | 3.8 | 3.7 | 3.5 |
| NIIRF (K=32) | seen       | 3.9 | 3.7 | 3.7 | 3.6 | 3.6 |
| Mag. NF      | unseen     | 4.9 | 4.7 | 4.7 | 4.6 | 4.6 |
| NIIRF (K=32) | unseen     | 4.2 | 3.9 | 3.9 | 3.8 | 3.7 |


## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.


## Copyright and license


Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```
Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

The following files:
* `pyscripts/models/hiir_downstream.py`
* `pyscripts/utils/extract_iir_params.py`

were adapted from https://github.com/yoyololicon/hrtf-notebooks (license included in [LICENSES/MIT.md](LICENSES/MIT.md)):

```
Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2023 Chin-Yun Yu
```
