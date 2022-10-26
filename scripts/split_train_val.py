import os
import shutil
import glob
import numpy as np
import math
import json
import trimesh
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenedir', type=str, help='input scene directory')
    parser.add_argument('--visualize', action="store_true", help='visualize camera positions with trimesh')
    parser.add_argument('--percent_train', type=float, default=80.0, help='percent of total pictures to be used for training')

    # REGNERF:
    # images_train
    # images_test

    args = parser.parse_args()
    TRAIN_RATIO = max(min(args.percent_train / 100.0, 1.0), 0.0)
    with open(os.path.join(args.scenedir, './transforms.json'),) as transforms_file:
        transforms = json.load(transforms_file)
        transforms_frames = transforms["frames"]
        N_transforms = len(transforms_frames)

        # path must end with / to make sure image path is relative
        if args.scenedir[-1] != '/':
            args.scenedir += '/'

        # TODO load images and split them
        # images = [f[len(opt.path):] for f in sorted(glob.glob(os.path.join(opt.path, opt.images, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]

        poses_bounds = np.load(os.path.join(args.scenedir, 'poses_bounds.npy'))
        N_pb = poses_bounds.shape[0]

        # print(f'[INFO] loaded {len(images)} images, {N} poses_bounds as {poses_bounds.shape}')

        # assert N == len(images)

        assert N_pb == N_transforms

        # train_mask = np.random.binomial(1, TRAIN_RATIO, size=N_pb).astype(bool)
        num_train = int(round(TRAIN_RATIO*N_pb))
        num_val = N_pb - num_train
        train_mask = np.array([True]*num_train + [False]*num_val)
        np.random.shuffle(train_mask)


        transforms_frames_train = [[transforms_frames[i] for i in range(len(transforms_frames)) if train_mask[i]]] 
        transforms_frames_val = [[transforms_frames[i] for i in range(len(transforms_frames)) if not train_mask[i]]] 
        
        transforms_train = dict(transforms)
        transforms_val = dict(transforms)
        transforms_train["frames"] = transforms_frames_train
        transforms_val["frames"] = transforms_frames_val

        with open(os.path.join(args.scenedir, './transforms_train.json'), "w") as outfile:
            json.dump(transforms_train, outfile, indent=2)
        with open(os.path.join(args.scenedir, './transforms_val.json'), "w") as outfile:
            json.dump(transforms_val, outfile, indent=2)

        poses_bound_train = poses_bounds[train_mask, ...]
        poses_bound_test = poses_bounds[~train_mask, ...]

        print(poses_bound_train.shape)
        print(poses_bound_test.shape)

        np.save(os.path.join(args.scenedir, 'poses_bounds_train.npy'), poses_bound_train)
        np.save(os.path.join(args.scenedir, 'poses_bounds_test.npy'), poses_bound_test)

        # IMAGES_DIR = args.images_dir
        # # potentially dangerous
        # try:
        #     shutil.rmtree(IMAGES_DIR)
        # except FileNotFoundError:
        #     pass
        # os.makedirs(IMAGES_DIR)