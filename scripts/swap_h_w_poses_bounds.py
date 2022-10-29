import os
import shutil
import glob
import numpy as np
import math
import json
import trimesh
import argparse

def delete_create_dir(dir):
    try:
        shutil.rmtree(dir)
    except FileNotFoundError:
        pass
    os.makedirs(dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='input scene directory')
    # parser.add_argument('--images_path', type=str, default='images', help="path for input images")

    args = parser.parse_args()

    # REGNERF:
    # images_train
    # images_test

    poses_bounds_npy = args.filename
    poses_bounds_backup_npy = args.filename+".old"


    if not os.path.exists(poses_bounds_npy):
        print("poses_bounds.npy not found in scenedir")
        exit(1)  

        # TODO load images and split them
        # images = [f for f in sorted(glob.glob(os.path.join(os.path.join(args.scenedir, args.images_path), "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]

    poses_bounds = np.load(poses_bounds_npy)
    N_pb = poses_bounds.shape[0]

    print("poses_bounds number: ", N_pb)

    print(poses_bounds[0])

    # print("N_pb == len(images): ", N_pb == len(images))

    poses_bounds = poses_bounds[:, [0,1,2,3,9,5,6,7,8,4,10,11,12,13,14,15,16]] # 0:15 matrix 15 16 bounds

    print(poses_bounds[0])

    shutil.copy(poses_bounds_npy, poses_bounds_backup_npy)

    np.save('poses_bounds.npy', poses_bounds)

    # IMAGES_DIR = args.images_dir
    # # potentially dangerous
    # try:
    #     shutil.rmtree(IMAGES_DIR)
    # except FileNotFoundError:
    #     pass
    # os.makedirs(IMAGES_DIR)