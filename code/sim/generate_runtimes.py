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

# prop_choices = [25, 50, 100, 200, 300, 500, 1000]
# # det_scales_choices = [ (2000, 720), (2000, 640), (2000, 600), (2000, 560), (2000, 480), (2000, 360) ]
# det_scales_choices = [ (2000, 720), (2000, 640), (2000, 600), (2000, 560), (2000, 480), (2000, 420), (2000, 360), (2000, 300), (2000, 240) ]

prop_choices = [100, 300, 500, 1000]
det_scales_choices = [ (2000, 600), (2000, 480), (2000, 420), (2000, 360), (2000, 300), (2000, 240) ]

model_name = opts.model_name
dataset = opts.dataset
model = DetectorWithRegressors(
        model_name,
        dataset,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        "cuda:0",
        None #models_info[dataset][model_name]["regressors"]
)

data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
)
seqs = data_loader.get_sequences_list() # [ sid ]
random.shuffle(seqs)
seqs = seqs[:3]

all_combs = itertools.product(
    prop_choices, det_scales_choices
)

_ = model.detect(np.zeros((1200, 1900, 3), np.uint8))

model_times = {}
for pr, det_scale in all_combs:
    model.change_num_proposals(pr)
    model.change_scale(det_scale)
    times = []
    for i, seq in enumerate(seqs):
        frames_info = data_loader.get_sequence(seq)
        for j, frame_info in enumerate(frames_info):
            image, bboxes, classes = data_loader.get_frame(frame_info["id"])

            t1 = perf_counter()

            if j % 30 == 0:
                result = model.detect(image, get_metrics=False, get_switch_metric=False, get_area_metric=False)
            else:
                result = model.detect(image)
            parsed_result = model.parse_result_for_sap(result)
            torch.cuda.synchronize()

            t2 = perf_counter()
            times.append(t2-t1)
    print(pr, det_scale[1], np.mean(times))
    model_times[pr, det_scale[1]] = copy.deepcopy(times)

out_path = os.path.join(opts.out_dir, "{}_{}.pkl".format(model_name, dataset))
with open(out_path, "wb") as f:
    pickle.dump(model_times, f)