#!/bin/bash
#PBS -N fus_defects_yolo_hyp
#PBS -o fus_defects_yolo_hyp.txt
#PBS -q gpu
#PBS -e fus_defects_yolo_hyp.txt
#PBS -k oe
#PBS -m e

#PBS -l select=1:ngpus=1:ncpus=12,walltime=12:00:00

NUM_NODES=$PBS_NUM_NODES_NODES
NUM_NODES=$(cat $PBS_NODEFILE | wc -l)

echo "start script"
#echo "entity": $model
#echo "data_cfg": $data_cfg

source $HOME/.bashrc
cd /davinci-1/home/dmor/artificial_intelligence/repos/fuselage_defects_detection

conda activate py38
python /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_detection/main.py --data_cfg $data_cfg --model $model
