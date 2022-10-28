#!/bin/bash

#SBATCH --job-name="nerf-pl-example"
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=16384

module load 2022r2
module load miniconda3

conda activate nerf_pl

DATASET=nuscenes-3
CKPT=epoch8.ckpt

python ../../papers/nerf_pl/eval.py --dataset_name llff --root_dir ../../../datasets/${DATASET}/llff --N_importance 64 --img_wh 1600 900 --split val