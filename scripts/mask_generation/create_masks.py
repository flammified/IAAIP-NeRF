# Setup detectron2 logger
import argparse
import glob
from pathlib import Path
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import torch
import os
import cv2
import sys
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# IDs for the following classes of interest: ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "bird", "cat", "dog"]
# Essentially, we only want to keep moving entities that may occur in the images
# see https://github.com/facebookresearch/detectron2/issues/147#issuecomment-645958806
CLASSES_OF_INTEREST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16]

# Get Model's Config and instantiate the predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

def create_mask(image, mask_filename):
    # Generate outputs
    outputs = predictor(image)

    # Only keep detected instances of our classes of interest
    instances = outputs["instances"]
    if instances.pred_classes.size(dim=0) != 0:
        instances = instances[torch.as_tensor([elem in CLASSES_OF_INTEREST for elem in instances.pred_classes])]

    # Use `Visualizer` to draw masks on the black image
    v = Visualizer(np.zeros_like(image[:, :, ::-1]), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    masks = instances.pred_masks.cpu().data.numpy()
    for m in masks:
        v.draw_binary_mask(m, color="w")
    output = v.get_output()

    # Save the freshly created mask
    mask_img = output.get_image()[:, :, ::-1]
    mask_img[np.nonzero(mask_img)] = 0xFF
    mask_img = cv2.GaussianBlur(mask_img,(5,5),0)
    cv2.imwrite(mask_filename, mask_img)

def create_masks(image_dir):

    filenames = [f for f in sorted(glob.glob(os.path.join(image_dir, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    N = len(filenames)
    print(f"Found {N} pictures in {image_dir}")
    print("Start creating masks for the specified images...")

    pbar = tqdm(total=len(filenames))

    for i, filename in enumerate(filenames):
        print(f"Image {i+1}/{N}")
        im = cv2.imread(filename)
        filename_without_extension = Path(filename).stem
        mask_filename = Path(image_dir).joinpath(f"dynamic_mask_{filename_without_extension}.png")
        create_mask(im, str(mask_filename))
        pbar.update(1)

    pbar.close()

    print("Masks successfully created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate a mask (with detectron2) for dynamic objects that can appear in car datasets, for each picture in the specified directories")

    parser.add_argument("image_dirs", nargs='+', type=str, help="directories containing the input images")

    args = parser.parse_args()
    for image_dir in args.image_dirs:
        # Create and save masks
        create_masks(image_dir)