import itertools
import random
import json
import numpy as np
import time

from data_loader import CustomCocoDetection, CustomCocoResult
from detector_with_regressors import DetectorWithRegressors
from dynamic_tracktor import Tracktor

from setup_info import *

model_names = [ "faster_rcnn" ]
det_scale_vals = [ (2000, 720) ] # [ (2000, 720), (2000, 640), (2000, 480), (2000, 320) ]
prop_vals = [ 500 ] # [100, 300, 500, 1000]

dataset = "argoverse"
# dataset = "imagenet_vid_argoverse_format"

device = "cuda:0"

data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
)

seqs = data_loader.get_sequences_list() # [ sid ]
random.shuffle(seqs)
# seqs = seqs[:50]
# print(seqs)
# exit()

all_combs = itertools.product(
    model_names, det_scale_vals, prop_vals
)

# def filter_func(x):
#     if x[1] == "ada" or x[2] == "ada":
#         return True
#     return x[1][1] >= x[2][1]
# all_combs = list(filter(lambda x: filter_func(x), all_combs))

all_combs_results = {}
for model_name, det_scale, pr in all_combs:

    det = DetectorWithRegressors(
        model_name,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        device,
        None,  # models_info[dataset][model_name]["regressors"],
    )
    det.change_scale(det_scale)
    det.change_num_proposals(pr)

    _ = det.detect(np.zeros((1200, 1900, 3), np.uint8))

    times = []
    results_obj = CustomCocoResult(dataset, dataset_info[dataset]["test_json"])

    for i, seq in enumerate(seqs):
        frames_info = data_loader.get_sequence(seq)
        seq_times = []
        for j, frame_info in enumerate(frames_info):
            image, bboxes, classes = data_loader.get_frame(frame_info["id"])
            ts = time.time()
            result = det.detect(image)
            seq_times.append((time.time() - ts) * 1000)
            results_obj.add_mmdet_results(frame_info, result)
        times.append(np.mean(seq_times))
        break
    results = results_obj.evaluate()
    key = "model_name_{}_det(sc)_{}".format(model_name, det_scale[1])
    all_combs_results[key] = {
        "average_time" : np.mean(times),
        "results" : results
    }
    print(key, np.mean(times), results)

print(all_combs_results)