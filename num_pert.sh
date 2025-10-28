#!/bin/bash

#PBS -N conv_3
#PBS -o Odin_002-.txt
#PBS -q gpu
#PBS -e Odin_002-.txt
#PBS -k oe
#PBS -m e
#PBS -M davide.mor@leonardo.com
#PBS -l select=1:ngpus=1:ncpus=4,walltime=150:00:00

# Add conda to source
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
# Conda activate
conda activate env_9


python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/test_time_data_augmentation.py --run_name "synbols_CSIx3" \
        --dataset "Synbols" \
        --cuda 0 \
        --seed 9 \
        --n_experiences 10 \
        --model "gresnet32" \
        --temperature 4 \
        --class_augmentation 3 \
        --with_rotations 1 \
        --n_aug 21
