import os
import glob
import numpy as np
import math
import json
import trimesh
import argparse

def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()

if __name__ == '__main__':
    with open("./transforms_example.json") as transforms_file:
        transforms = json.load(transforms_file)

        # TODO copy all pictures to /images dir and name them sequentially

        frames = []

        for f in transforms["frames"]:
            frame = np.array(f)
            frames.append(frame)

        print(frames[0])

        for frame in frames:
            h, w = frame.w, frame.h
            f = ( frame.fl_x + frame.fl_y ) / 2.0
            # w, h, f = factor * w, factor * h, factor * f
            hwf = np.array([h,w,f]).reshape([3,1])

            # inverse function of colmap2nerf
            c2w[2, :] *= -1 # flip whole world upside down
            c2w = c2w[[1, 0, 2, 3], :] # swap y and z
            c2w[ 0:3, 1] *= -1
            c2w[0:3, 2] *= -1

            # for k in imdata:
            #     im = imdata[k]
            #     R = im.qvec2rotmat()
            #     t = im.tvec.reshape([3,1])
            #     m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            #     w2c_mats.append(m)
    
            # w2c_mats = np.stack(w2c_mats, 0)
            # c2w_mats = np.linalg.inv(w2c_mats)
            
            # poses = c2w_mats[:, :3, :4].transpose([1,2,0])
            poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)


            # from /LLFF/llff/poses/pose_utils.py
            # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
            poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)



        # # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
        # poses = np.concatenate([poses[:, 1, :], poses[:, 0, :], -poses[:, 2, :], poses[:, 3, :], poses[:, 4, :]], 1)



        # VISUALIZATION

        # # load data
        # #images = [f[len(opt.path):] for f in sorted(glob.glob(os.path.join(opt.path, opt.images, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]

        # #poses_bounds = np.load(os.path.join(opt.path, 'poses_bounds.npy'))
        # #N = poses_bounds.shape[0]

        # #print(f'[INFO] loaded {len(images)} images, {N} poses_bounds as {poses_bounds.shape}')

        # #assert N == len(images)

        # poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
        # bounds = poses_bounds[:, -2:] # (N, 2)

        # H, W, fl = poses[0, :, -1] 

        # H = H // opt.downscale
        # W = W // opt.downscale
        # fl = fl / opt.downscale

        # print(f'[INFO] H = {H}, W = {W}, fl = {fl} (downscale = {opt.downscale})')

        # # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
        # poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)

        # # to homogeneous 
        # last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
        # poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

        # visualize_poses(poses)