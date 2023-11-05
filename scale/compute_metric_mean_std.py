import itertools
import random
import json
import numpy as np
import time

from data_loader import CustomCocoDetection
from detector_with_regressors import DetectorWithRegressors
from dynamic_tracktor import Tracktor

from setup_info import *

model_name = "faster_rcnn"
dataset = "argoverse"
# dataset = "imagenet_vid_argoverse_format"
device = "cuda:3"
tracker_name = "tracktor_faster_rcnn"

data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
)
det = DetectorWithRegressors(
    model_name,
    models_info[dataset][model_name]["config_file"],
    models_info[dataset][model_name]["checkpoint_file"],
    device,
    models_info[dataset][model_name]["regressors"],
)
det.change_scale((2000, 720))
seqs = data_loader.get_sequences_list() # [ sid ]

vals = []

for i, seq in enumerate(seqs):
    frames_info = data_loader.get_sequence(seq)
    for j, frame_info in enumerate(frames_info):
        if j % 15 == 0:
            image, bboxes, classes = data_loader.__getitem__(i)
            meta = data_loader.__getmeta__(i)
            result, metrics = det.detect(image, get_metrics=True)
            # vals.append(metrics[2:10].copy())
            print(metrics)
            break
    break

# print(vals)
# print(np.max(vals, axis=0))
# print(np.std(vals, axis=0))