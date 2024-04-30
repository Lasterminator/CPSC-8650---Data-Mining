#!/bin/bash
#
#PBS -N trainer
#PBS -l select=1:ncpus=16:mpiprocs=16:ngpus=1:gpu_model=a100:mem=64gb:interconnect=hdr,walltime=8:00:00
#PBS -o trainer_consistent.txt
#PBS -j oe

nvidia-smi
cd $PBS_O_WORKDIR
module load anaconda3/2022.05-gcc/9.5.0 cuda/11.8.0-gcc/9.5.0 cudnn/8.7.0.84-11.8-gcc/9.5.0 libpng/1.6.37-gcc/9.5.0
source activate pytorch-a100
python train_model.py