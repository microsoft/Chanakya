import itertools
import random
import json
import numpy as np
import time

from data_loader import CustomCocoDetection, CustomCocoResult
from detector_with_regressors import DetectorWithRegressors
from dynamic_tracktor import Tracktor

from setup_info import *

model_name = "faster_rcnn"
dataset = "argoverse"
# dataset = "imagenet_vid_argoverse_format"
device = "cuda:0"
tracker_name = "tracktor_faster_rcnn"

data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
)
det = DetectorWithRegressors(
    model_name,
    models_info[dataset][model_name]["config_file"],
    models_info[dataset][model_name]["checkpoint_file"],
    device,
    "resnet_50",
    None,  # models_info[dataset][model_name]["regressors"],
)
if tracker_name.split("_")[0] == "tracktor":
    tracker = Tracktor(
        tracker_name,
        tracker_info[dataset][tracker_name]["config_file"],
        tracker_info[dataset][tracker_name]["checkpoint_file"],
        device,
        # regressor_config=tracker_info[dataset][tracker_name]["regressors"]
    )
else:
    raise ("Invalid Tracker!!!")

seqs = data_loader.get_sequences_list() # [ sid ]
random.shuffle(seqs)

proposal_vals = [50, 100, 300, 500]
det_scale_vals = [(2000, 720), (2000, 600), (2000, 480)]
tracker_scale_vals = [(2000, 720), (2000, 600), (2000, 480), (2000, 360)]
time_strides = [5, 10, 15]

# proposal_vals = [100]  # [ 100, 300, 500, 1000 ]
# det_scale_vals = [(2000, 600), (2000, 480), (2000, 360), (2000, 240)]
# tracker_scale_vals = [(2000, 600), (2000, 480), (2000, 360)]
# time_strides = [5, 15]

all_combs = itertools.product(
    proposal_vals, det_scale_vals, tracker_scale_vals, time_strides
)
def filter_func(x):
    if x[1] == "ada" or x[2] == "ada":
        return True
    return x[1][1] >= x[2][1]
all_combs = list(filter(lambda x: filter_func(x), all_combs))

all_combs_results = {}
for pr, det_scale, tracker_scale, time_stride in all_combs:

    det.change_num_proposals(pr)
    if det_scale is not "ada":
        det.change_scale(det_scale)
    if tracker_scale is not "ada":
        tracker.change_scale(tracker_scale)


    times = []
    results_obj = CustomCocoResult(dataset, dataset_info[dataset]["test_json"])

    for i, seq in enumerate(seqs):
        frames_info = data_loader.get_sequence(seq)
        seq_times = []
        prev_result = None
        for j, frame_info in enumerate(frames_info):
            image, bboxes, classes = data_loader.get_frame(frame_info["id"])
            ts = time.time()
            if j == 0 or j % time_stride == 0:
                result = det.detect(image)
            else:
                try:
                    result = tracker.track(image, result)
                except:
                    result = prev_result  # copy prev result in worst case if tracker fails
            seq_times.append((time.time() - ts) * 1000)
            results_obj.add_mmdet_results(frame_info, result)
            prev_result = result
        times.append(np.mean(seq_times))

    results = results_obj.evaluate()
    key = "pr_{}_det(sc)_{}_tra(sc)_{}_stride_{}".format(pr, det_scale[1], tracker_scale[1], time_stride)
    all_combs_results[key] = {
        "average_time" : np.mean(times),
        "results" : results
    }
    print(key, np.mean(times), results)

result_fname = "tradeoff_results_{}_{}_{}.json".format(dataset, model_name, tracker_name)
with open(result_fname, "w") as f:
    json.dump(all_combs_results, f)