#!/bin/bash

#PBS -N tiny_CA1
#PBS -o exp.txt
#PBS -q gpu
#PBS -e exp.txt
#PBS -k oe
#PBS -m e
#PBS -M davide.mor@leonardo.com
#PBS -l select=1:ngpus=1:ncpus=12,walltime=240:00:00

# Add conda to source
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
# Conda activate
conda activate env_9

class_augmentation=2  #multiplier of the number of classes
rotations=0           #1 if we want to include test time data augmentation, 0 otherwise
n_aug=1              #maximal number of test time data augmentation in which we are interested in
train_model=0         #0 if the model have already been trained and do not want to train it again
n_seed=1              #number of seeds in which we train each experiment

run_name="tiny_CA${class_augmentation}"

if [ "$train_model" -eq 1 ]; then
  for seed in $(seq 0 $((n_seed-1)))
  do
      python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/main.py --run_name $run_name \
              --dataset "TinyImageNet" \
              --cuda 0 \
              --seed $seed \
              --n_experiences 10 \
              --model "gresnet32" \
              --epochs 100 \
              --lr 0.005 \
              --scheduler 70 90 \
              --epochs_distillation 120 \
              --lr_distillation 0.035 \
              --scheduler_distillation 80 110 \
              --temperature 12 \
              --class_augmentation $class_augmentation
  done
fi

python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/test_time_data_augmentation.py --run_name $run_name \
        --dataset "TinyImageNet" \
        --cuda 0 \
        --seed $((n_seed-1))\
        --n_experiences 10 \
        --model "gresnet32" \
        --temperature 12 \
        --class_augmentation $class_augmentation \
        --with_rotations $rotations \
        --n_aug $n_aug