#!/bin/sh
#SBATCH --job-name=instantngprender
#SBATCH --partition=gpu
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G

export EXPERIMENT_DIR=./data/nerf/nuscenes6_6cam
./train_instant-ngp.sh

export CAMERA_PATH_DIR=../IAAIP-NeRF/ablation/camera_paths/instant-ngp
export CAMERA_PATH_NAME=nuscenes6_groundtruth_lane
./render_instant-ngp.sh
export CAMERA_PATH_NAME=nuscenes6_unseen_lane
./render_instant-ngp.sh
export CAMERA_PATH_NAME=nuscenes6_zigzag
./render_instant-ngp.sh


