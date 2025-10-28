#!/bin/bash

#PBS -N synbols_all_2
#PBS -o exp.txt
#PBS -q gpu
#PBS -e exp.txt
#PBS -k oe
#PBS -m e
#PBS -M davide.mor@leonardo.com
#PBS -l select=1:ngpus=1:ncpus=4,walltime=240:00:00

# Add conda to source
source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
# Conda activate
conda activate env_9

train_model=0         #0 if the model have already been trained and do not want to train it again
n_seed=10             #number of seeds in which we train each experiment

for class_augmentation in 1 2 3 4
do
  run_name="synbols_CA${class_augmentation}"
  if [ "$train_model" -eq 1 ]; then
    for seed in $(seq 0 $((n_seed-1)))
    do
        python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/main.py --run_name $run_name \
                --dataset "Synbols" \
                --cuda 0 \
                --seed $seed \
                --n_experiences 10 \
                --model "gresnet32" \
                --epochs 25 \
                --lr 0.005 \
                --scheduler 10 20 \
                --epochs_distillation 25 \
                --lr_distillation 0.035 \
                --scheduler_distillation 15 \
                --temperature 4 \
                --class_augmentation $class_augmentation
    done
  fi
done


for class_augmentation in 1 2 3 4
do
  for rotations in 0 1
  do
    run_name="synbols_CA${class_augmentation}"
    if ! [[ "$class_augmentation" -eq 1 && "$rotations" -eq 1 ]]; then
      if [ "$rotations" -eq 1 ]; then
        n_aug=$((class_augmentation * 2))
      else
        n_aug=2
      fi

      python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/test_time_data_augmentation.py --run_name $run_name \
              --dataset "Synbols" \
              --cuda 0 \
              --seed $((n_seed-1))\
              --n_experiences 10 \
              --model "gresnet32" \
              --temperature 4 \
              --class_augmentation $class_augmentation \
              --with_rotations $rotations \
              --n_aug $n_aug
    fi
  done
done