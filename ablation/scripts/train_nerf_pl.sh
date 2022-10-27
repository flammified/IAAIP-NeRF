#!/bin/bash

#SBATCH --job-name="nerf-pl-example"
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16384

module load 2022r2
module load miniconda3

conda activate nerf_pl

DATASET=nuscenes-3

python ../../papers/nerf_pl/train.py --dataset_name llff --root_dir ../../../datasets/${DATASET}/llff --N_importance 64 --img_wh 1600 900 --num_epochs 30 --batch_size 512 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 --exp_name ${DATASET}