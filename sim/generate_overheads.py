import argparse, json, pickle
from time import perf_counter, time
import itertools
import random
import os, copy

import numpy as np
import torch

from chanakya.detector_with_regressors import DetectorWithRegressors
from chanakya.data_loader import CustomCocoDetection, CustomCocoResult
# from chanakya.dynamic_tracktor import Tracktor
from chanakya.setup_info import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model-name', type=str, required=True)
opts = parser.parse_args()


model_name = opts.model_name
dataset = opts.dataset
model = DetectorWithRegressors(
        model_name,
        dataset,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        "cuda:0",
        models_info[dataset][model_name]["regressors"]
)

data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
)
seqs = data_loader.get_sequences_list() # [ sid ]
random.shuffle(seqs)
seqs = seqs[:1]

prop = 500
det_scales = [ (2000, 1200), (2000, 900), (2000, 600), (2000, 300) ]
# strides = [15, 30, 60]

model.change_num_proposals(prop)


_ = model.detect(np.zeros((1200, 1900, 3), np.uint8))

model_times_metrics = {}
for det_scale in det_scales:
    model.change_scale(det_scale)
    times = []
    for i, seq in enumerate(seqs):
        frames_info = data_loader.get_sequence(seq)
        for j, frame_info in enumerate(frames_info):
            image, bboxes, classes = data_loader.get_frame(frame_info["id"])

            t1 = perf_counter()
            result, metrics = model.detect(image, predict_scale=True) #get_metrics=True, get_switch_metric=False, get_area_metric=True)
            parsed_result = model.parse_result_for_sap(result)
            torch.cuda.synchronize()
            t2 = perf_counter()

            times.append(t2-t1)
    model_times_metrics[det_scale[1]] = copy.deepcopy(times)
    print(det_scale[1], np.round(np.mean(times)*1000, 3) )

# model_times_no_metrics = {}
# for det_scale in det_scales:
#     model.change_scale(det_scale)
#     times = []
#     for i, seq in enumerate(seqs):
#         frames_info = data_loader.get_sequence(seq)
#         for j, frame_info in enumerate(frames_info):
#             image, bboxes, classes = data_loader.get_frame(frame_info["id"])

#             t1 = perf_counter()
#             result = model.detect(image)            
#             parsed_result = model.parse_result_for_sap(result)
#             torch.cuda.synchronize()
#             t2 = perf_counter()

#             times.append(t2-t1)
#     model_times_no_metrics[det_scale[1]] = copy.deepcopy(times)
#     print(det_scale[1], np.round(np.mean(times)*1000, 3) )
