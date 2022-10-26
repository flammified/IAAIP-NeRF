#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from gettext import translation
import os
from pathlib import Path, PurePosixPath
from tkinter import Y
from traceback import print_tb

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

from nuscenes.nuscenes import NuScenes


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--video_in", default="",
                        help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
    parser.add_argument("--video_fps", default=2)
    parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")
    # parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
    parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive", "sequential", "spatial", "transitive",
                        "vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--colmap_db", default="colmap.db",
                        help="colmap database filename")
    parser.add_argument("--colmap_camera_model", default="OPENCV", choices=[
                        "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"], help="camera model")
    parser.add_argument("--colmap_camera_params", default="",
                        help="intrinsic parameters, depending on the chosen model.  Format: fx,fy,cx,cy,dist")
    parser.add_argument("--images", default="images",
                        help="input path to the images")
    parser.add_argument("--text", default="colmap_text",
                        help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--aabb_scale", default=16, choices=["1", "2", "4", "8", "16"],
                        help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--skip_early", default=0,
                        help="skip this many images from the start")
    parser.add_argument("--keep_colmap_coords", action="store_true",
                        help="keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering)")
    parser.add_argument("--out", default="transforms.json", help="output path")
    parser.add_argument("--vocab_path", default="",
                        help="vocabulary tree path")
    parser.add_argument('--totp', nargs='+', type=float, help='center of attention. Subtracted from the position of each camera.')
    parser.add_argument('--up', nargs='+', type=float, help='up vector. Used to reorient the scene.')
    parser.add_argument('--avglen', type=float, help='avg distance of cameras from center. Used as an inverse scaling factor.')
    parser.add_argument("--adaptive_rescale",  action="store_true", help="compute totp and avglen automatically, instead of using the ones provided via --totp --avglen")
    parser.add_argument("--num_dataset_samples", required=True, type=int, help="number of samples (car positions) to be loaded from the dataset")
    parser.add_argument("--scene_num", required=True, type=int, help="id of the scene to be loaded")
    parser.add_argument("--sensors", nargs="+", choices=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"], help="sensors (cameras) to be used", default=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"])
    parser.add_argument("--nuscenes_dataroot", default="/data/sets/nuscenes", help="default is /data/sets/nuscenes")
    args = parser.parse_args()
    return args


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def run_ffmpeg(args):
    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)
    images = "\"" + args.images + "\""
    video = "\"" + args.video_in + "\""
    fps = float(args.video_fps) or 1.0
    print(
        f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)
    try:
        # Passing Images' Path Without Double Quotes
        shutil.rmtree(args.images)
    except:
        pass
    do_system(f"mkdir {images}")

    time_slice_value = ""
    time_slice = args.time_slice
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"
    do_system(
        f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec) -> np.ndarray:
    return np.array([
        [
                    1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
                    2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                    2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
                    ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db):
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom


def process_nuscenes_sample(sample, sensor_keys=["CAM_FRONT"]):
    print({k: sample[k] for k in ('token', 'next', 'timestamp')})
    for sensor in sensor_keys:
        cam_data = nusc.get('sample_data', sample['data'][sensor])
        # {'token', 'sample_token', 'ego_pose_token', 'calibrated_sensor_token', 'timestamp': 1535385095904799, 'fileformat': 'jpg', 'is_key_frame': True, 'height': 900, 'width': 1600, 'filename': 'samples/CAM_FRONT_LEFT/n008-2018-08-27-11-48-51-0400__CAM_FRONT_LEFT__1535385095904799.jpg', 'prev', 'next', 'sensor_modality': 'camera', 'channel': 'CAM_FRONT_LEFT'}
        print(sensor)

        abs_path = os.path.join(nusc.dataroot, cam_data["filename"])
        # image_rel = os.path.relpath(IMAGE_FOLDER)
        # name = str(f"./{image_rel}/{'_'.)}")
        b = sharpness(abs_path)
        print(abs_path, "sharpness=", b)

        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        # could be cached
        calibrated_sensor = nusc.get(
            'calibrated_sensor', cam_data['calibrated_sensor_token'])
        print(ego_pose)

        print(np.array(calibrated_sensor["rotation"]))

        ego_rotation = qvec2rotmat(np.array(ego_pose["rotation"]))
        ego_translation = np.array(ego_pose["translation"]).reshape([3, -1])
        camera_rotation = qvec2rotmat(np.array(calibrated_sensor["rotation"]))
        camera_translation = np.array(
            calibrated_sensor["translation"]).reshape([3, -1])

        print(ego_rotation)
        # print(camera_rotation)

        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        
        # rotation = camera_rotation @ ego_rotation
        # translation = camera_translation + ego_translation
        
        ego: np.ndarray = np.concatenate( [np.concatenate([ego_rotation, ego_translation], 1), bottom], 0)
        camera: np.ndarray = np.concatenate( [np.concatenate([camera_rotation, camera_translation], 1), bottom], 0)


        # translation = np.array([0,0,0]).reshape([3,-1])

        # ego : np.ndarray = np.concatenate([np.concatenate([ego_rotation, ego_translation], 1), bottom], 0)
        # camera : np.ndarray = np.concatenate([np.concatenate([camera_rotation, camera_translation], 1), bottom], 0)

        # print(rotation)
        # print(translation)

        # c2w: np.ndarray = np.concatenate( [np.concatenate([rotation, translation], 1), bottom], 0)
        c2w = ego @ camera

        conv_mat = np.array([
            [0,  1, 0, 0],
            [-1, 0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
        c2w[2, :] *= -1  # flip whole world upside down

        # c2w = c2w @ conv_mat

        #  [[1252.8131021185304, 0.0, 826.588114781398], [0.0, 1252.8131021185304, 469.9846626224581], [0.0, 0.0, 1.0]]}
        camera_intrinsic = np.array(calibrated_sensor["camera_intrinsic"])

        # undistorted pictures
        k1 = k2 = p1 = p2 = 0

        fl_x = camera_intrinsic[0, 0]
        fl_y = camera_intrinsic[1, 1]
        #s = camera_intrinsic[0,1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]

        w = cam_data["width"]
        h = cam_data["height"]

        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2
        fovx = angle_x * 180 / math.pi
        fovy = angle_y * 180 / math.pi

        print(
            f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

        # TODO cache
        camera_out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h
        }
        # print(c2w)
        frame = {"file_path":	os.path.relpath(
            abs_path), "sharpness": b, "transform_matrix": c2w, **camera_out}
        yield frame


if __name__ == "__main__":
    args = parse_args()
    nusc = NuScenes(version='v1.0-mini', dataroot=args.nuscenes_dataroot, verbose=True)
    SCENE_SCALE_COEFF = 0.5

    args = parse_args()
    SENSOR_KEYS = args.sensors
    NUM_SAMPLES = args.num_dataset_samples
    SCENE_NUM = args.scene_num # 3 or 6
    
    NUM_SAMPLES = 14
    SCENE_NUM = 3 # 6
    SCENE_SCALE_COEFF = 0.5

    if args.video_in != "":
        run_ffmpeg(args)

    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    IMAGE_FOLDER = args.images
    TEXT_FOLDER = args.text
    OUT_PATH = args.out

    ADAPTIVE_RESCALE = args.adaptive_rescale
    TOTP = args.totp
    AVGLEN = args.avglen
    UP=args.up

    if not ADAPTIVE_RESCALE:
        if not TOTP or len(TOTP)!=3:
            print("Please specify --totp x y z, or use --adaptive_rescale")
            exit(1)
        else:
            print(TOTP)
            TOTP = np.array(TOTP)
            print(TOTP.shape)
        if not AVGLEN:
            print("Please specify --avglen, or use --adaptive_rescale")
            exit(1)
        if not UP or len(UP)!=3:
            print("Please specify --up, or use --adaptive_rescale")
            exit(1)


    print(f"outputting to {OUT_PATH}...")

    out = {
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }

    my_scene = nusc.scene[SCENE_NUM]
    sample_token = my_scene['first_sample_token']
    # uncomment to visualize scene
    # nusc.render_sample(sample_token)
    # exit()
    sample = nusc.get('sample', sample_token)
    for frame in process_nuscenes_sample(sample, SENSOR_KEYS):
        out["frames"].append(frame)
    for i in range(min(NUM_SAMPLES-1, my_scene["nbr_samples"])):
        sample_token = sample["next"]
        sample = nusc.get('sample', sample_token)
        for frame in process_nuscenes_sample(sample, SENSOR_KEYS):
            out["frames"].append(frame)

    if ADAPTIVE_RESCALE:
        up = np.zeros(3)

        for f in out["frames"]:
            up += f["transform_matrix"][0:3,1]

        # reorient the scene to be easier to work with
        up = up / np.linalg.norm(up)
        print("up vector was", up)

    else:
        up = UP

    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(
            R, f["transform_matrix"])  # rotate up to be the z axis

    if ADAPTIVE_RESCALE:
        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(
                    mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print("totp:", totp)  # the cameras are looking at totp
    else:
        totp = TOTP

    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    if ADAPTIVE_RESCALE:
        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= len(out["frames"])
        print("avg camera distance from origin", avglen)
        
    else:
        avglen = AVGLEN

    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4.0 * SCENE_SCALE_COEFF / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)