import json

import mmcv
import numpy as np
import cv2

from setup_info import *
from data_loader import CustomCocoDetection, CustomCocoResult
from detector_with_regressors import DetectorWithRegressors

def draw_bboxes(img, bboxes, out_name, extra=None, bbox_color='green', text_color='green'):
    if extra is None:
        extra = np.array([str(i + 1) for i in range(bboxes.shape[0])])
    mmcv.imshow_det_bboxes(img, bboxes, extra, show=False, out_file=out_name, bbox_color=bbox_color, text_color=text_color)

dataset = "argoverse"
test_data_loader = CustomCocoDetection(
    dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset,
)
model_name = "faster_rcnn"
det = DetectorWithRegressors(
    model_name,
    models_info[dataset][model_name]["config_file"],
    models_info[dataset][model_name]["checkpoint_file"],
    'cuda:3',
    None,  # models_info[dataset][model_name]["regressors"],
)

seq = 17
frames_info = test_data_loader.get_sequence(seq)
frame_info = frames_info[50]

image, bboxes, classes = test_data_loader.get_frame(frame_info["id"])

result = det.detect(image) 
det.detector.show_result(image, result, out_file="temp2.jpg")
# result = det.parse_result_for_sap(result)
res_bboxes, res_classes = det.parse_result_for_sap(result)
# print(result)

res_file = "/FatigueDataDrive/HAMS-Edge-Datasets/Exp/Argoverse-HD/output/frcnn50_s0.75/val/results.json"

# with open(res_file) as f:
#     res_js = json.load(f)

# res_bboxes = [ r["bbox"] for r in res_js if r["image_id"] == frame_info["id"] ]
# res_classes = [ r["category_id"] for r in res_js if r["image_id"] == frame_info["id"] ]
# for r in res_bboxes:
#     r[2] += r[0]
#     r[3] += r[1]
# res_bboxes = np.array(res_bboxes)
# res_classes = np.array(res_classes)
# print(res_classes)

bboxes = bboxes.cpu().numpy()
classes = classes.cpu().numpy()
draw_bboxes(image.copy(), bboxes, "temp.jpg", extra=classes)
image = mmcv.imread("temp.jpg")
draw_bboxes(image.copy(), res_bboxes, "temp.jpg", extra=res_classes, bbox_color='red', text_color='red')
