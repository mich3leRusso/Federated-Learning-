#!/bin/bash

#PBS -N cifar100_all_plotaug
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

n_aug=50               #maximal number of test time data augmentation in which we are interested in
train_model=0          #0 if the model have already been trained and do not want to train it again
n_seed=10              #number of seeds in which we train each experiment
control=0              #use the rotations as positive examples instead of using them as negative examples
control_2=0            #do not use augmented classes but train with them anyway
control_ttda=0


for class_augmentation in 1
do
  if [ "$control" -eq 1 ]; then
    run_name="cifar100_control_CA${class_augmentation}"
  elif [ "$control_ttda" -ne 0 ]; then
    run_name="cifar100_control_ttda${control_ttda}_CA${class_augmentation}"
  else
    run_name="cifar100_CA${class_augmentation}"
  fi


  if [ "$train_model" -eq 1 ]; then
    for seed in $(seq 0 $((n_seed-1)))
    do
        python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/main.py --run_name $run_name \
                --dataset "CIFAR100" \
                --cuda 0 \
                --seed $seed \
                --n_experiences 10 \
                --model "gresnet32" \
                --epochs 50 \
                --lr 0.005 \
                --scheduler 35 \
                --epochs_distillation 50 \
                --lr_distillation 0.035 \
                --scheduler_distillation 40 \
                --temperature 6.5 \
                --class_augmentation $class_augmentation\
                --control $control \
                --control_ttda $control_ttda

    done
  fi
done

for class_augmentation in 1 2 3 4
do
  for rotations in 0 1
  do
    if [ "$control" -eq 1 ]; then
      run_name="cifar100_control_CA${class_augmentation}"
    elif [ "$control_ttda" -ne 0 ]; then
      run_name="cifar100_control_ttda${control_ttda}_CA${class_augmentation}"
    else
      run_name="cifar100_CA${class_augmentation}"
    fi

    if ! [[ "$class_augmentation" -eq 1 && "$rotations" -eq 1 ]]; then
      python /davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/test_time_data_augmentation.py --run_name $run_name \
              --dataset "CIFAR100" \
              --cuda 0 \
              --seed $((n_seed-1))\
              --n_experiences 10 \
              --model "gresnet32" \
              --temperature 6.5 \
              --class_augmentation $class_augmentation \
              --with_rotations $rotations \
              --n_aug $n_aug \
              --control $control \
              --control_2 $control_2 \
              --control_ttda $control_ttda
    fi
  done
done
