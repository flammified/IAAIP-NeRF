module load 2022r2
module load cuda/11.6
module load miniconda3

SCENE_NUM=3

mkdir -p /scratch/$USER/datasets/nuscenes
wget -q -O- https://www.nuscenes.org/data/v1.0-mini.tgz | tar -xz -C /scratch/$USER/datasets/nuscenes/

# we are using our own fork of https://github.com/NVlabs/instant-ngp/
git clone git@github.com:Duico/instant-ngp.git /scratch/$USER/Duico_instant-ngp

# follow instant-ngp install steps:
# conda create -p /scratch/$USER/conda_instantngp python=3.8
# pip install -r requirements.txt
# cmake . -B build -DNGP_BUILD_WITH_GUI=off
# cmake --build build --config RelWithDebInfo -j

pip install nuscenes-devkit
conda activate /scratch/aduico/conda_instantngp/

cd /scratch/aduico/Duico_instant-ngp
EXPERIMENT_DIR=data/nerf/nuscenes${SCENE_NUM}_6cam
mkdir -p $EXPERIMENT_DIR # change folder depending on scene and cameras used, i.e. ./nuscenes3_1front  ./nuscenes3_3front  ./nuscenes3_6cam
cd $EXPERIMENT_DIR

# run this once per scene to obtain a fixed value for --up, --totp, --avglen
../../../scripts/dataset2nerf_nuscenes.py --nuscenes_dataroot /scratch/aduico/datasets/nuscenes/ --aabb_scale=4 --adaptive_rescale --num_dataset_samples 14 --scene_num $SCENE_NUM # without --sensors it will use all available cameras
# this will output something like:
# up vector was [ 0.01769267 -0.00154715 -0.99984228]
# computing center of attention...
# totp: [-541.36052319 1923.06989148   14.16387316]
# avg camera distance from origin 13.998675314165954

# run each time in a different folder, changing --sensors
../../../scripts/dataset2nerf_nuscenes.py --nuscenes_dataroot /scratch/aduico/datasets/nuscenes/ --aabb_scale=4 --num_dataset_samples 14 --scene_num $SCENE_NUM --up 0.01769267 -0.00154715 -0.99984228 --totp -541.36052319 1923.06989148 14.16387316 --avglen 13.998675314165954 --sensors CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_RIGHT CAM_BACK_LEFT

cd /scratch/aduico/Duico_instant-ngp
# on GPU
python ./scripts/run.py  --save_snapshot ${EXPERIMENT_DIR}/base.msgpack  --mode nerf --n_steps 30000 --scene ${EXPERIMENT_DIR}/

CAMERA_PATH_DIR=../IAAIP-NeRF/ablation/camera_paths/instant-ngp
CAMERA_PATH_NAME=nuscenes3_unseen_lane
# on GPU
python scripts/run.py --mode nerf --scene ${EXPERIMENT_DIR} --load_snapshot ${EXPERIMENT_DIR}/base.msgpack --video_camera_path ${CAMERA_PATH_DIR}/${CAMERA_PATH_NAME}.json --video_output ${EXPERIMENT_DIR}/${CAMERA_PATH_NAME}.mp4 --video_n_seconds=20 --video_fps=25