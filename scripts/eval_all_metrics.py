import imageio
import numpy as np
import os
import struct

def mse2psnr(x):
	return -10.*np.log(x)/np.log(10.)

def write_image_imageio(img_file, img, quality):
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:,:,:3]
		kwargs["quality"] = quality
		kwargs["subsampling"] = 0
	imageio.imwrite(img_file, img, **kwargs)

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

	raise ValueError(f"Unknown metric: {metric}.")

def compute_error(metric, img, ref):
	metric_map = compute_error_img(metric, img, ref)
	metric_map[np.logical_not(np.isfinite(metric_map))] = 0
	if len(metric_map.shape) == 3:
		metric_map = np.mean(metric_map, axis=2)
	mean = np.mean(metric_map)
	return mean


if __name__ == '__main__':
    
    print("Evaluating test transforms from ", args.test_transforms)
    with open(args.test_transforms) as f:
        test_transforms = json.load(f)
    data_dir=os.path.dirname(args.test_transforms)
    totmse = 0
    totpsnr = 0
    totssim = 0
    totcount = 0
    minpsnr = 1000
    maxpsnr = 0

    # Evaluate metrics on black background
    testbed.background_color = [0.0, 0.0, 0.0, 1.0]

    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4

    testbed.fov_axis = 0
    testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
    testbed.shall_train = False

    # with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
    for p in images_test:
        p = frame["file_path"]
        ref_fname = os.path.join(data_dir, p)
        if not os.path.isfile(ref_fname):
            ref_fname = os.path.join(data_dir, p + ".png")
            if not os.path.isfile(ref_fname):
                ref_fname = os.path.join(data_dir, p + ".jpg")
                if not os.path.isfile(ref_fname):
                    ref_fname = os.path.join(data_dir, p + ".jpeg")
                    # if not os.path.isfile(ref_fname):
                    # 	ref_fname = os.path.join(data_dir, p + ".exr")

        ref_image = read_image(ref_fname)

        # testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
        # image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)
        
        #TODO
        #image = # load the same filename from the images_snapshot folder
         

        diffimg = np.absolute(image - ref_image)
        diffimg[...,3:4] = 1.0

        A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
        R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
        mse = float(compute_error("MSE", A, R))
        ssim = float(compute_error("SSIM", A, R))
        totssim += ssim
        totmse += mse
        psnr = mse2psnr(mse)
        totpsnr += psnr
        minpsnr = psnr if psnr<minpsnr else minpsnr
        maxpsnr = psnr if psnr>maxpsnr else maxpsnr
        totcount = totcount+1
        t.set_postfix(psnr = totpsnr/(totcount or 1))

    psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    psnr = totpsnr/(totcount or 1)
    ssim = totssim/(totcount or 1)
    print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")