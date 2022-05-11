#!/bin/sh

#PBS -q a6500g10q@vm-pbs2
#PBS -l select=1:ngpus=1:ncpus=1:mem=16g
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -m abe

echo PBS_O_WORKDIR $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

source /opt/shared/anaconda/anaconda3-2020/bin/activate
conda activate akboldinov

ls -l
/usr/bin/nvidia-smi -L
echo
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

python -u train_multitask_wav2vec_golos.py /mnt/scratch/ws/akboldinov/202206251205ws2/data models/wav2vec2-large-xlsr-53 64
echo 1