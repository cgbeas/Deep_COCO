#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov 20 20:05:24 2017

@author: Carlos Beas
"""

import numpy as np
import random
import ijson
from pycocotools.coco import COCO
import json



def getCocoDataSplit(dataset, split, af):
    """
    This function partitions a COCO dataset by number of images.
    It will return a subset of 'dataset' which will be equivalent to
    the fraction of the dataset represented by 'split'.  'af' is the path to 'dataset'
    The funtcion returns two dictionaries:  The first containing a sample of size defined by split
    of the original dataset, and the second is the original dataset minus the sample.
    
    <dataset>:  This corresponds to the original and entire dataset.  Usualy in the form of a json or dictionary
    <split>:  This is a floating point number indicating the fraction of the dataset that will be used to create the sample
    <af>:  This is the path to the json file for the original dataset.
    """
    
    image_sample = {}
    val_minusSample = {}
    
    print('Loading: ', af)
    coco_caps=COCO(af)
    sample_size = int(len(dataset['images']) * split)
    print("Sample has size ", sample_size)
    indexes = np.arange(0, len(dataset['images']))
    sample_val = np.random.choice(indexes, size=sample_size, replace=False)
    
    not_val = set(indexes) - set(sample_val)
    not_val = np.array(list(not_val))
    
    print(len(not_val))
    
    #image_sample = dataset['images'][np.where(dataset['images'] in sample_val)]
    image_sample['images'] = dataset['images'][:sample_size]
    val_minusSample['images'] = dataset['images'][sample_size:]
    #val_minusSample = dataset['images'][dataset['images'] == not_val]
    
    my_anns = []
    for img in image_sample['images']:
        annIds = coco_caps.getAnnIds(imgIds=img['id']);
        anns = coco_caps.loadAnns(annIds)
        #my_anns[img['id']] = anns
        for ann in anns:
            my_anns.append(ann)

    image_sample['annotations'] = my_anns
    
    
    my_anns = []
    for img in val_minusSample['images']:
        annIds = coco_caps.getAnnIds(imgIds=img['id']);
        anns = coco_caps.loadAnns(annIds)
        #my_anns[img['id']] = anns
        for ann in anns:
            my_anns.append(ann)

    val_minusSample['annotations'] = my_anns
    
    with open('./mini_val2017.json', 'w') as f:
        json.dump(image_sample, f)
        f.close()
        
    with open('./val_minusSample2017.json', 'w') as f:
        json.dump(val_minusSample, f)
        f.close()
    
    return image_sample, val_minusSample

# annFile = '../../../data/2017/annotations_stuff/stuff_val2017.json'
mini_val, train_val = getCocoDataSplit(val_data, 0.1, annFile)
mini_val['annotations'][0]