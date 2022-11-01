#!/bin/bash

#SBATCH --job-name="monodepth2"
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16384

module load 2022r2
module load miniconda3

conda activate monodepth2

DATASET=nuscenes-3
MODEL=mono_1024x320

python ../../../monodepth2/test_simple.py --image_path ../../../datasets/${DATASET}/images --model_name ${MODEL}


