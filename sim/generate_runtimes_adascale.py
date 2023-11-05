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
parser.add_argument('--out-dir', type=str, required=True)
opts = parser.parse_args()

det_scales_choices = [ (2000, x) for x in range(240, 610, 10) ]

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

_ = model.detect(np.zeros((1200, 1900, 3), np.uint8))

model_times = {}
for det_scale in det_scales_choices:
    model.change_scale(det_scale)
    times = []
    for i, seq in enumerate(seqs):
        frames_info = data_loader.get_sequence(seq)
        for j, frame_info in enumerate(frames_info):
            image, bboxes, classes = data_loader.get_frame(frame_info["id"])

            t1 = perf_counter()

            result = model.detect(image, predict_scale=True) #, get_metrics=True)
            
            # parsed_result = model.parse_result_for_sap(result)
            torch.cuda.synchronize()

            t2 = perf_counter()
            times.append(t2-t1)
    print(det_scale[1], np.mean(times))
    model_times[det_scale[1]] = copy.deepcopy(times)

out_path = os.path.join(opts.out_dir, "{}_{}_adascale.pkl".format(model_name, dataset))
with open(out_path, "wb") as f:
    pickle.dump(model_times, f)