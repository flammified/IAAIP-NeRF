#!/bin/sh
#SBATCH --job-name=instantngprender
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G

EXPERIMENT_DIR=./data/nerf/nuscenes3_1front
CAMERA_PATH_DIR=../IAAIP-NeRF/ablation/camera_paths/instant-ngp
# CAMERA_PATH_NAME=nuscenes3_groundtruth_lane_01
CAMERA_PATH_NAME=nuscenes3_unseen_lane

module load 2022r2
module load cuda/11.6
# module load python
# module load py-pip
module load miniconda3

conda activate /scratch/aduico/conda_instantngp

echo "EXPERIMENT_DIR: ${EXPERIMENT_DIR}"
echo "CAMERA_PATH_NAME: ${CAMERA_PATH_NAME}"

python scripts/run.py --mode nerf --scene ${EXPERIMENT_DIR} --load_snapshot ${EXPERIMENT_DIR}/base.msgpack --video_camera_path ${CAMERA_PATH_DIR}/${CAMERA_PATH_NAME}.json --video_output ${EXPERIMENT_DIR}/${CAMERA_PATH_NAME}.mp4 --video_n_seconds=20 --video_fps=25
