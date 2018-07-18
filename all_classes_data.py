
import sys
sys.path.append("PythonAPI/")
from PythonAPI.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


dataDir='/media/reynaldo/Data/Databases/coco/coco_2017'
dataType = {'train':'train2017', 'val':'val2017'}

annThingsFile='%s/annotations/instances_%s.json'%(dataDir,dataType["val"])
annStuffFile='%s/annotations/stuff_%s.json'%(dataDir,dataType["val"])


# initialize COCO api for instance annotations
coco_things=COCO(annThingsFile)
coco_stuff=COCO(annStuffFile)

# display COCO categories and supercategories
thing_cats = coco_things.loadCats(coco_things.getCatIds())
stuff_cats = coco_stuff.loadCats(coco_stuff.getCatIds())


# Things Classes # 

nms=[cat['name'] for cat in thing_cats]
print('\nCOCO Things Categories: \n\n', ' '.join(nms))

nms = set([cat['supercategory'] for cat in thing_cats])
print('\nCOCO Things Supercategories: \n', ' '.join(nms))

things_count = 0
for cat in thing_cats:
	things_count += 1

print("\nThing classes total: ", things_count)

# Stuff Classes # 

nms=[cat['name'] for cat in stuff_cats]
print('\nCOCO Stuff Categories: \n\n', ' '.join(nms))

nms = set([cat['supercategory'] for cat in stuff_cats])
print('\nCOCO Stuff Supercategories: \n', ' '.join(nms))

stuff_count = 0
for cat in stuff_cats:
	stuff_count += 1

print("\nStuff classes total: ", stuff_count)


print("\nTotal number of things and stuff classes:", things_count + stuff_count, "\n")



