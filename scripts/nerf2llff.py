import os
import shutil
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
    parser = argparse.ArgumentParser()
    parser.add_argument('scenedir', type=str, help='input scene directory')
    parser.add_argument('--images_dir', type=str, default='images', help="images folder (do not include full path, e.g., just use `images_4`)")
    parser.add_argument('--visualize', action="store_true", help='visualize camera positions with trimesh')

    args = parser.parse_args()

    with open(os.path.join(args.scenedir, './transforms.json'),) as transforms_file:
        transforms = json.load(transforms_file)
        
        IMAGES_DIR = os.path.join(args.scenedir, args.images_dir)
        # potentially dangerous
        try:
            shutil.rmtree(IMAGES_DIR)
        except FileNotFoundError:
            pass
        os.makedirs(IMAGES_DIR)

        poses_bounds_list = []

        for i, frame in enumerate(transforms["frames"]):
            # TODO copy all pictures to /images dir and name them sequentially
            file_path = frame["file_path"]

            # TODO use same file extension
            destination_path = os.path.join(IMAGES_DIR, f"image_{i:04d}.jpg")
            shutil.copyfile(os.path.join(args.scenedir,file_path), destination_path)

            w = frame["w"] if "w" in frame else transforms["w"]
            h = frame["h"] if "h" in frame else transforms["h"]
            fl_x = frame["fl_x"] if "fl_x" in frame else transforms["fl_x"]
            fl_y = frame["fl_y"] if "fl_y" in frame else transforms["fl_y"]
            f = ( fl_x + fl_y ) / 2.0
            # w, h, f = factor * w, factor * h, factor * f
            hwf = np.array([h,w,f]).reshape([3,1])

            c2w = np.array(frame["transform_matrix"])

            # inverse function of colmap2nerf (moved below)
            # c2w[2, :] *= -1 # flip whole world upside down
            # c2w = c2w[[1, 0, 2, 3], :] # swap y and z
            # c2w[0:3, 1] *= -1
            # c2w[0:3, 2] *= -1

            # remove bottom row ([0 0 0 1])
            c2w = c2w[0:3, :]

            # inverse function of colmap2nerf
            c2w[2, :] *= -1 # flip whole world upside down
            c2w = c2w[[1, 0, 2], :] # swap y and z
            c2w[:, 1] *= -1
            c2w[:, 2] *= -1

            # SOURCE: /LLFF/llff/poses/pose_utils.py, colmap -> llff
            # for k in imdata:
            #     im = imdata[k]
            #     R = im.qvec2rotmat()
            #     t = im.tvec.reshape([3,1])
            #     m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            #     w2c_mats.append(m)
    
            # w2c_mats = np.stack(w2c_mats, 0)
            # c2w_mats = np.linalg.inv(w2c_mats)
            
            # poses = c2w_mats[:, :3, :4].transpose([1,2,0])
            # poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)

            pose = np.concatenate([c2w, hwf], 1)

            # SOURCE: /LLFF/llff/poses/pose_utils.py
            # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
            pose = np.concatenate((pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]), 1)

            pose = pose.flatten()
            bounds = np.array([0.1, 10])

            pose_bounds = np.concatenate((pose, bounds))

            poses_bounds_list.append(pose_bounds)

        poses_bounds = np.vstack(poses_bounds_list)

        # TODO set bounds properly
        # for k in pts3d:
        # pts_arr.append(pts3d[k].xyz)
        # cams = [0] * poses.shape[-1]
        # for ind in pts3d[k].image_ids:
        #     if len(cams) < ind - 1:
        #         print('ERROR: the correct camera poses for current points cannot be accessed')
        #         return
        #     cams[ind-1] = 1
        # vis_arr.append(cams)

        #     pts_arr = np.array(pts_arr)
        #     vis_arr = np.array(vis_arr)
        #     print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
            
        #     zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
        #     valid_z = zvals[vis_arr==1]
        #     print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
            
        #     save_arr = []
        #     for i in perm:
        #         vis = vis_arr[:, i]
        #         zs = zvals[:, i]
        #         zs = zs[vis==1]
        #         close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        #         # print( i, close_depth, inf_depth )
                
        #         save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))            
        
        np.save(os.path.join(args.scenedir, 'poses_bounds.npy'), poses_bounds)


        # # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
        # poses = np.concatenate([poses[:, 1, :], poses[:, 0, :], -poses[:, 2, :], poses[:, 3, :], poses[:, 4, :]], 1)



        # VISUALIZATION

        # # load data
        # #images = [f[len(opt.path):] for f in sorted(glob.glob(os.path.join(opt.path, opt.images, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]

        # #poses_bounds = np.load(os.path.join(opt.path, 'poses_bounds.npy'))
        # #N = poses_bounds.shape[0]

        # #print(f'[INFO] loaded {len(images)} images, {N} poses_bounds as {poses_bounds.shape}')

        # #assert N == len(images)

        if args.visualize: 
            poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
            bounds = poses_bounds[:, -2:] # (N, 2)

            H, W, fl = poses[0, :, -1] 

            # H = H // opt.downscale
            # W = W // opt.downscale
            # fl = fl / opt.downscale

            print(f'[INFO] H = {H}, W = {W}, fl = {fl} )')

            # # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
            poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)

            # # to homogeneous 
            last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
            poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

            visualize_poses(poses)