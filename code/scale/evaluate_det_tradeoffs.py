import itertools
import random
import json
import numpy as np
import time
import copy

from data_loader import CustomCocoDetection, CustomCocoResult
from detector_with_regressors import DetectorWithRegressors
from dynamic_tracktor import Tracktor

from setup_info import *

model_names = [ "yolov3", "fcos" ]
det_scale_vals = [(2000, 480), (2000, 384), (2000, 320)]

# dataset = "argoverse"
dataset = "imagenet_vid_argoverse_format"

device = "cuda:0"

# data_loader = CustomCocoDetection(
#     dataset_info[dataset]["train_root"], dataset_info[dataset]["train_json"], dataset
# )

# seqs = data_loader.get_sequences_list() # [ sid ]
# random.shuffle(seqs)


with open(dataset_info[dataset]["train_json"]) as f:
    data_loader = json.load(f)

arr = data_loader["images"]
lens = []
curr_seq = -1
curr_len = 0

i = 0
while i < len(arr):
    curr_sid = arr[i]["sid"]
    curr_idx = i
    while (i < len(arr)) and (arr[i]["sid"] == curr_sid):
        i += 1
    lens.append((curr_sid, data_loader['sequences'][curr_sid], i - curr_idx))

# print(lens)
# lens = [ len(data_loader.get_sequence(x)) for x in seqs ]
# print(len(list(filter(lambda x: x[-1] > 100, lens))))
# print(len(list(filter(lambda x: x[-1] > 300, lens))))
# print(len(list(filter(lambda x: x[-1] > 500, lens))))
# print(len(list(filter(lambda x: x[-1] > 1000, lens))))
# print(len(list(filter(lambda x: x[-1] > 2000, lens))))

seqs_100 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 100, copy.deepcopy(lens))) ]
print(len(seqs_100))
seqs_300 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 300, copy.deepcopy(lens))) ]
print(len(seqs_300))
seqs_500 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 500, copy.deepcopy(lens))) ]
print(len(seqs_500))
seqs_1000 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 1000, copy.deepcopy(lens))) ]
print(len(seqs_1000))

seqs_400_600 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 400 and a[-1] < 600, copy.deepcopy(lens))) ]
print(len(seqs_400_600))

seqs_300_700 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 300 and a[-1] < 700, copy.deepcopy(lens))) ]
print(len(seqs_300_700))

seqs_500_1000 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 500 and a[-1] < 1000, copy.deepcopy(lens))) ]
print(len(seqs_500_1000))

seqs_700_1000 = [ [x[0], x[1]] for x in list(filter(lambda a: a[-1] > 700 and a[-1] < 1000, copy.deepcopy(lens))) ]
print(len(seqs_700_1000))

with open("imagenet_vid_seqs.json", "w") as f:
    json.dump({
        "seqs_greater_100" : seqs_100,
        "seqs_greater_300" : seqs_300,
        "seqs_greater_500" : seqs_500,
        "seqs_greater_1000" : seqs_1000,
        "seqs_400_600" : seqs_400_600,
        "seqs_300_700" : seqs_300_700,
        "seqs_500_1000" : seqs_500_1000,
        "seqs_700_1000" : seqs_700_1000
    }, f)

exit()
# seqs = seqs[:50]
# print(seqs)
# exit()

all_combs = itertools.product(
    model_names, det_scale_vals
)

# def filter_func(x):
#     if x[1] == "ada" or x[2] == "ada":
#         return True
#     return x[1][1] >= x[2][1]
# all_combs = list(filter(lambda x: filter_func(x), all_combs))

all_combs_results = {}
for model_name, det_scale in all_combs:

    det = DetectorWithRegressors(
        model_name,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        device,
        None,  # models_info[dataset][model_name]["regressors"],
    )
    det.change_scale(det_scale)

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
    results = results_obj.evaluate()
    key = "model_name_{}_det(sc)_{}".format(model_name, det_scale[1])
    all_combs_results[key] = {
        "average_time" : np.mean(times),
        "results" : results
    }
    print(key, np.mean(times), results)

print(all_combs_results)