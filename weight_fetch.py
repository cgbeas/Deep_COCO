import model as modellib
import sys
sys.path.append("PythonAPI/")

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


if __name__ == '__main__':

	class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()


	model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)

	model_path = model.get_imagenet_weights()
	print("Loading weights ", model_path)
	model.load_weights(model_path, by_name=True)
	# Validation dataset
	dataset_val = CocoDataset()
	coco = dataset_val.load_coco(args.dataset, "minival", return_coco=True)
	dataset_val.prepare()

	# TODO: evaluating on 500 images. Set to 0 to evaluate on all images.
	evaluate_coco(dataset_val, coco, "bbox", limit=500)