"""

Name: plot_train_Data.py
Author:  Carlos Beas
Date Last Modified:  11/25/2017

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

%matplotlib inline

log_file = './stuff_train_log.txt'

metrics_dict = {
    'loss' : [],
    'rpn_class_loss' : [],
    'rpn_bbox_loss' : [],
    'mrcnn_class_loss' : [],
    'mrcnn_bbox_loss' : [],
    'mrcnn_mask_loss' : []
}

val_metrics_dict = {
    'loss' : [],
    'rpn_class_loss' : [],
    'rpn_bbox_loss' : [],
    'mrcnn_class_loss' : [],
    'mrcnn_bbox_loss' : [],
    'mrcnn_mask_loss' : [],
    'val_loss' : [],
    'val_rpn_class_loss' : [],
    'val_rpn_bbox_loss' : [],
    'val_mrcnn_class_loss' : [],
    'val_mrcnn_bbox_loss' : [],
    'val_mrcnn_mask_loss' : []
}

with open(log_file, 'r') as file:
    for line in file:
        #loss: 3.8000
        matches = re.findall('(\d+\.\d+)', line)
        
        if(len(matches) == 6):
            metrics_dict['loss'].append(float(matches[0]))
            metrics_dict['rpn_class_loss'].append(float(matches[1]))
            metrics_dict['rpn_bbox_loss'].append(float(matches[2]))
            metrics_dict['mrcnn_class_loss'].append(float(matches[3]))
            metrics_dict['mrcnn_bbox_loss'].append(float(matches[4]))
            metrics_dict['mrcnn_mask_loss'].append(float(matches[5]))
        elif(len(matches) == 12):
            val_metrics_dict['loss'].append(float(matches[0]))
            val_metrics_dict['rpn_class_loss'].append(float(matches[1]))
            val_metrics_dict['rpn_bbox_loss'].append(float(matches[2]))
            val_metrics_dict['mrcnn_class_loss'].append(float(matches[3]))
            val_metrics_dict['mrcnn_bbox_loss'].append(float(matches[4]))
            val_metrics_dict['mrcnn_mask_loss'].append(float(matches[5]))
            val_metrics_dict['val_loss'].append(float(matches[6]))
            val_metrics_dict['val_rpn_class_loss'].append(float(matches[7]))
            val_metrics_dict['val_rpn_bbox_loss'].append(float(matches[8]))
            val_metrics_dict['val_mrcnn_class_loss'].append(float(matches[9]))
            val_metrics_dict['val_mrcnn_bbox_loss'].append(float(matches[10]))
            val_metrics_dict['val_mrcnn_mask_loss'].append(float(matches[11]))


r = np.arange(0, len(metrics_dict['mrcnn_mask_loss']))
print(len(r))
print(len(metrics_dict['loss']))

analysis_df = pd.DataFrame(metrics_dict, index=r)
print(analysis_df.head())

analysis_df.plot()

r = np.arange(0, len(val_metrics_dict['val_loss']))
print(len(r))
print(len(val_metrics_dict['val_loss']))

val_analysis_df = pd.DataFrame(val_metrics_dict, index=r)
print(val_analysis_df.head())

val_analysis_df.plot()


