import time
import random
import copy
import itertools
import numpy as np

from setup_info import *
from data_loader import CustomCocoDetection, CustomCocoResult
from detector_with_regressors import DetectorWithRegressors

model_name = "faster_rcnn"
device = "cuda:3"
dataset = "imagenet_vid_argoverse_format"

test_data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"],
    dataset_info[dataset]["test_json"],
    dataset,
    argoverse_extensions=True,
)

det = DetectorWithRegressors(
    model_name,
    models_info[dataset][model_name]["config_file"],
    models_info[dataset][model_name]["checkpoint_file"],
    device,
    "resnet_50",
    None,  # models_info[dataset][model_name]["regressors"],
)

# seqs = test_data_loader.get_sequences_list() # [ sid ]
# random.shuffle(seqs)
# seqs = seqs[:20]

# easy_seqs = [7020, 7022, 7024, 7026, 8000, 8002, 9000, 9001, 15000, 16000, 16013, 18000, 18001, 19002, 23012, 23013, 26001, 26002, 29002, 30000, 35000]
# hard_seqs = [122000, 77000, 4, 128001, 99001, 31001, 33002, 16007, 17001, 105000, 165000, 118001, 129000, 81000, 0, 171000, 2, 118009, 161001, 13001, 33001]

easy_seqs2 = [
    4000,
    5003,
    7020,
    7024,
    7026,
    8000,
    8002,
    9000,
    9001,
    11003,
    12000,
    12002,
    15000,
    15002,
    16000,
    16013,
    18000,
    18001,
    19002,
]
hard_seqs2 = [
    99001,
    128001,
    33002,
    171000,
    16007,
    129000,
    105000,
    118009,
    2,
    16006,
    76002,
    138000,
    158000,
    31000,
    116000,
    39002,
    75002,
    7005,
    89000,
    133004,
    133000,
]
seqs = copy.deepcopy(easy_seqs2)

proposal_vals = [100]  # [50, 100, 300]
scale_vals = [(608, 608), (416, 416), (320, 320)]
# scale_vals = [(2000, 600), (2000, 480), (2000, 360), (2000, 240)]
for pr, scale in itertools.product(proposal_vals, scale_vals):

    det.change_num_proposals(pr)
    det.change_scale(scale)

    times = []
    results_obj = CustomCocoResult(dataset, dataset_info[dataset]["test_json"])

    for i, seq in enumerate(seqs):
        frames_info = test_data_loader.get_sequence(seq)
        seq_times = []
        for j, frame_info in enumerate(frames_info):
            image, bboxes, classes = test_data_loader.get_frame(frame_info["id"])
            data = det.preprocess_image(image)
            ts = time.time()
            result = det(data)
            te = (time.time() - ts) * 1000
            seq_times.append(te)
            # det.change_scale((2000, new_scale))
            results_obj.add_mmdet_results(frame_info, result)
        times.append(np.mean(seq_times[1:]))

    print(pr, scale, np.mean(times))
    results_obj.evaluate()

# det = DetectorWithRegressors(
#     model_name,
#     models_info[dataset][model_name]["config_file"],
#     models_info[dataset][model_name]["checkpoint_file"],
#     device,
#     "resnet_50",
#     models_info[dataset][model_name]["regressors"],
# )

# import matplotlib
# import matplotlib.pyplot as plt

# def plot_scale_frame(scales, orig_scale, save_name):
#     frame_nums = [i for i in range(len(scales))]
#     plt.plot(frame_nums, scales)
#     plt.savefig(save_name)
#     plt.close()

# proposal_vals = [50, 100, 300] # [25, 50, 100, 300]
# for pr in proposal_vals:

#     det.change_num_proposals(pr)

#     times = []
#     results_obj = CustomCocoResult(dataset, dataset_info[dataset]["test_json"])

#     for i, seq in enumerate(seqs):
#         det.change_scale((2000, 600))
#         frames_info = test_data_loader.get_sequence(seq)
#         seq_times = []
#         scales = []
#         for j, frame_info in enumerate(frames_info):
#             image, bboxes, classes = test_data_loader.get_frame(frame_info["id"])
#             data = det.preprocess_image(image)
#             ts = time.time()
#             result, new_scale = det(data)
#             te = (time.time() - ts) * 1000
#             seq_times.append(te)
#             det.change_scale((2000, new_scale))
#             scales.append(new_scale)
#             results_obj.add_mmdet_results(frame_info, result)
#         times.append(np.mean(seq_times[1:]))
#         plot_scale_frame(scales, frames_info[0]["height"], "plots/seq_{}.png".format(seq))

#     print(pr, "ada", np.mean(times))
#     results_obj.evaluate()
