#!/bin/sh
#SBATCH --job-name=instantngptrain
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16G

EXPERIMENT_DIR=./data/nerf/nuscenes3_1front

module load 2022r2
module load cuda/11.6
# module load python
# module load py-pip
module load miniconda3

conda activate /scratch/aduico/conda_instantngp

echo "EXPERIMENT_DIR: ${EXPERIMENT_DIR}"
python ./scripts/run.py  --save_snapshot ${EXPERIMENT_DIR}/base.msgpack  --mode nerf --n_steps 30000 --scene ${EXPERIMENT_DIR}/
