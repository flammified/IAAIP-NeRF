#!/bin/bash

#SBATCH --job-name="nerf-pl-example"
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16384

module load 2022r2
module load miniconda3

conda activate monodepth2

python ../../../monodepth2/test_simple.py --image_path ../../../datasets/nuscenes-3/images --model_name mono_1024x320


