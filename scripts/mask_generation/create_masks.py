# Setup detectron2 logger
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

    # Use `Visualizer` to draw blue masks on the black image
    v = Visualizer(np.zeros_like(image[:, :, ::-1]), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    masks = instances.pred_masks.cpu().data.numpy()
    for m in masks:
        v.draw_binary_mask(m, color="b")
    output = v.get_output()

    # Save the freshly created mask
    mask_file = output.get_image()[:, :, ::-1]
    cv2.imwrite(mask_filename, mask_file)

def create_masks(image_dir):
    print("Start creating masks for the specified images...")

    filenames = os.listdir(image_dir)
    pbar = tqdm(total=len(filenames))

    for filename in filenames:
        im = cv2.imread(os.path.join(image_dir, filename))
        filename_without_extension = filename[:-4]
        mask_filename = f'{image_dir}/dynamic_mask_{filename_without_extension}.png'
        create_mask(im, mask_filename)
        pbar.update(1)

    pbar.close()

    print("Masks successfully created!")

if __name__ == "__main__":
    # Create and save masks
    image_dir = sys.argv[1]
    create_masks(image_dir)