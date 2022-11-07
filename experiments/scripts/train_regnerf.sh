#!/bin/sh
#SBATCH --job-name=hanerftrain
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load 2022r2
module load cuda/11.6
# module load python
# module load py-pip
module load miniconda3

conda activate ha_nerf

python train_mask_grid_sample.py \
  --root_dir ../datasets/kitti360-static \
  --dataset_name blender \
  --image_wh 1408 376
  --save_dir ../results/kitti360-static/chck \
  --use_cache \
  --num_epochs 50 \
  --N_vocab 250 \
  --exp_name kitti360-static \
  --num_gpus 1 \
  --useNeuralRenderer false
