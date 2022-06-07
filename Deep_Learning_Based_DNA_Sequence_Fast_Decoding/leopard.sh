#!/bin/bash
#PBS -N Leopard_NCI
#PBS -P il82
#PBS -r y
#PBS -q gpuvolta  
#PBS -l storage=gdata/ik06 
#PBS -l walltime=01:00:00 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=80GB
#PBS -M ke.ding@anu.edu.au
#PBS -m e


###########################
#load modules for gpu support
module load cuda
module load cudnn
module load nccl
module load openmpi

# setup conda environment 
# -- change the path to your own conda directory
source /g/data/ik06/stark/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate leopard

# run the bechmark over one GPUs
# -- change the path to your own 
source /g/data/ik06/stark/NCI_Leopard/train.sh

