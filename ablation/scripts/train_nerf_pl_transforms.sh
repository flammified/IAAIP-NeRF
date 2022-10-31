#!/bin/bash

#SBATCH --job-name="nerf-pl"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=128G

module load 2022r2
module load miniconda3

conda activate nerf_pl

DATASET=nuscenes-3

python ../../papers/nerf_pl/train.py --dataset_name transforms --root_dir ../../../datasets/${DATASET} --N_importance 64 --img_wh 1600 900 --num_epochs 30 --batch_size 512 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 --exp_name ${DATASET} --num_gpus 2