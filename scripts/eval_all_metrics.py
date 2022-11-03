import imageio
import numpy as np
import os
from pathlib import Path
import struct
import argparse
import glob
from scipy.ndimage.filters import convolve1d
from tqdm import tqdm
import lpips
import torch

def mse2psnr(x): return -10.*np.log(x)/np.log(10.)

def read_image_imageio(img_file):
    img = imageio.imread(img_file)
    img = np.asarray(img).astype(np.float32)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    return img / 255.0

def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def read_image(file):
    if os.path.splitext(file)[1] == ".bin":
        with open(file, "rb") as f:
            bytes = f.read()
            h, w = struct.unpack("ii", bytes[:8])
            img = np.frombuffer(bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
    else:
        img = read_image_imageio(file)
        if img.shape[2] == 4:
            img[...,0:3] = srgb_to_linear(img[...,0:3])
            # Premultiply alpha
            img[...,0:3] *= img[...,3:4]
            #CUSTOM: discard alpha channel
            img = img[...,0:3]
        else:
            img = srgb_to_linear(img)
    return img

def write_image(file, img, quality=95):
    if os.path.splitext(file)[1] == ".bin":
        if img.shape[2] < 4:
            img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
        with open(file, "wb") as f:
            f.write(struct.pack("ii", img.shape[0], img.shape[1]))
            f.write(img.astype(np.float16).tobytes())
    else:
        if img.shape[2] == 4:
            img = np.copy(img)
            # Unmultiply alpha
            img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
            img[...,0:3] = linear_to_srgb(img[...,0:3])
        else:
            img = linear_to_srgb(img)
        write_image_imageio(file, img, quality)

def trim(error, skip=0.000001):
    error = np.sort(error.flatten())
    size = error.size
    skip = int(skip * size)
    return error[skip:size-skip].mean()

def luminance(a):
    a = np.maximum(0, a)**0.4545454545
    return 0.2126 * a[:,:,0] + 0.7152 * a[:,:,1] + 0.0722 * a[:,:,2]

def SSIM(a, b):
    def blur(a):
        k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
        x = convolve1d(a, k, axis=0)
        return convolve1d(x, k, axis=1)
    a = luminance(a)
    b = luminance(b)
    mA = blur(a)
    mB = blur(b)
    sA = blur(a*a) - mA**2
    sB = blur(b*b) - mB**2
    sAB = blur(a*b) - mA*mB
    c1 = 0.01**2
    c2 = 0.03**2
    p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
    p2 = (2.0*sAB + c2)/(sA + sB + c2)
    error = p1 * p2
    return error

def L1(img, ref):
    return np.abs(img - ref)

def APE(img, ref):
    return L1(img, ref) / (1e-2 + ref)

def SAPE(img, ref):
    return L1(img, ref) / (1e-2 + (ref + img) / 2.)

def L2(img, ref):
    return (img - ref)**2

def RSE(img, ref):
    return L2(img, ref) / (1e-2 + ref**2)

def rgb_mean(img):
    return np.mean(img, axis=2)

def compute_error_img(metric, img, ref):
    img[np.logical_not(np.isfinite(img))] = 0
    img = np.maximum(img, 0.)
    if metric == "MAE":
        return L1(img, ref)
    elif metric == "MAPE":
        return APE(img, ref)
    elif metric == "SMAPE":
        return SAPE(img, ref)
    elif metric == "MSE":
        return L2(img, ref)
    elif metric == "MScE":
        return L2(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
    elif metric == "MRSE":
        return RSE(img, ref)
    elif metric == "MtRSE":
        return trim(RSE(img, ref))
    elif metric == "MRScE":
        return RSE(np.clip(img, 0, 100), np.clip(ref, 0, 100))
    elif metric == "SSIM":
        return SSIM(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
    elif metric in ["FLIP", "\FLIP"]:
        # Set viewing conditions
        monitor_distance = 0.7
        monitor_width = 0.7
        monitor_resolution_x = 3840
        # Compute number of pixels per degree of visual angle
        pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

        ref_srgb = np.clip(flip.color_space_transform(ref, "linrgb2srgb"), 0, 1)
        img_srgb = np.clip(flip.color_space_transform(img, "linrgb2srgb"), 0, 1)
        result = flip.compute_flip(flip.utils.HWCtoCHW(ref_srgb), flip.utils.HWCtoCHW(img_srgb), pixels_per_degree)
        assert np.isfinite(result).all()
        return flip.utils.CHWtoHWC(result)

    raise ValueError(f"Unknown metric: {metric}.")

def compute_error(metric, img, ref):
    metric_map = compute_error_img(metric, img, ref)
    metric_map[np.logical_not(np.isfinite(metric_map))] = 0
    if len(metric_map.shape) == 3:
        metric_map = np.mean(metric_map, axis=2)
    mean = np.mean(metric_map)
    return mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="usage: python ./scripts/eval_all_metrics.py --images_test $SCENE_DIR/images_test --images_rendered $SCENE_DIR/images_screenshot")
    parser.add_argument('--images_test', type=str, default='images_test', help="path to ground truth images")
    parser.add_argument('--images_rendered', type=str, default='images_screenshot', help="path to rendered images")
    args = parser.parse_args()

    lpips_loss = lpips.LPIPS(net='alex')

    print("Computing metrics between ", args.images_rendered, " and ", args.images_test)
    # with open(args.test_transforms) as f:
    #     test_transforms = json.load(f)
    # data_dir=os.path.dirname(args.test_transforms)
    totmse = 0
    totpsnr = 0
    totssim = 0
    totlpips = 0
    totcount = 0
    minpsnr = 1000
    maxpsnr = 0

    images_test = [f for f in glob.glob(os.path.join(args.images_test, "*")) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]

    # with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
    with tqdm(images_test, unit="images", desc="Computing metrics") as t:
        for image_test_path in t:
            image_path_noext = os.path.join(args.images_rendered, Path(image_test_path).stem)

            ext_list = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".exr"]

            image_path = None
            for ext in ext_list:
                image_path = image_path_noext + ext
                if os.path.isfile(image_path):
                    break
            

            # ref_fname = os.path.join(data_dir, p)
            # if not os.path.isfile(ref_fname):
            #     ref_fname = os.path.join(data_dir, p + ".png")
            #     if not os.path.isfile(ref_fname):
            #         ref_fname = os.path.join(data_dir, p + ".jpg")
            #         if not os.path.isfile(ref_fname):
            #             ref_fname = os.path.join(data_dir, p + ".jpeg")
                        # if not os.path.isfile(ref_fname):
                        # 	ref_fname = os.path.join(data_dir, p + ".exr")

            if not os.path.exists(image_path):
                print("No matching rendered image")
                exit(1)

            ref_image = read_image(image_test_path)
            image = read_image(image_path)   

            diffimg = np.absolute(image - ref_image)
            diffimg[...,3:4] = 1.0

            A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
            R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
            mse = float(compute_error("MSE", A, R))
            ssim = float(compute_error("SSIM", A, R))
            # Nx3xHxW
            A_torch = torch.from_numpy(A).permute(2,0,1).unsqueeze(0)
            R_torch = torch.from_numpy(R).permute(2,0,1).unsqueeze(0)
        
            lpips_d = lpips_loss.forward(A_torch,R_torch).item()
            totssim += ssim
            totmse += mse
            totlpips += lpips_d
            psnr = mse2psnr(mse)
            totpsnr += psnr
            minpsnr = psnr if psnr<minpsnr else minpsnr
            maxpsnr = psnr if psnr>maxpsnr else maxpsnr
            totcount = totcount+1
            t.set_postfix(psnr = totpsnr/(totcount or 1), lpips=totlpips / (totcount or 1))
            # exit()

    psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    psnr = totpsnr/(totcount or 1)
    ssim = totssim/(totcount or 1)
    lpips_d = totlpips/(totcount or 1)
    print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim} LPIPS={lpips_d}")