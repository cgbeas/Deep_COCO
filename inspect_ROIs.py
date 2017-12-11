import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

#%matplotlib inline 

# Root directory of the project
#ROOT_DIR = os.getcwd()
ROOT_DIR = '/Users/carlosbeas/Desktop/Deep_Learning/COCO_Project/repos/Deep_COCO_repo/Deep-COCO'
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, '/Users/carlosbeas/Desktop/Deep_Learning/COCO_Project/repos/Deep_COCO_repo/Deep-COCO')

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")  #mask_rcnn_coco_0039.h5 #mask_rcnn_coco_0025_rois1.h5
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_0025_rois1.h5")


# Path to Shapes trained weights
#SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")

PYCOCO_DIR = '/Users/carlosbeas/Desktop/Deep_Learning/COCO_Project/repos/Deep_COCO_repo/Deep-COCO/PythonAPI'
sys.path.append(PYCOCO_DIR)

TOOLS_DIR = '/Users/carlosbeas/Desktop/Deep_Learning/COCO_Project/repos/Deep_COCO_repo/Deep-COCO/PythonAPI/pycocotools'
sys.path.append(TOOLS_DIR)

# MS COCO Dataset
import coco_stuff as coco
config = coco.CocoConfig()
COCO_DIR = "/Users/carlosbeas/Desktop/Deep_Learning/COCO_Project/data/2017"

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Build validation dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "minival")

# Must call before using the dataset
dataset.prepare()
print("Images: {}".format(len(dataset.image_ids)))


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set weights file path
if config.NAME == "shapes":
    weights_path = SHAPES_MODEL_PATH
elif config.NAME == "coco":
    weights_path = COCO_MODEL_PATH
# Or, uncomment to load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#image_id = random.choice(dataset.image_ids)
image_id = 48
print(image_id)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)




