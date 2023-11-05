from cv2 import data
import numpy as np
from numpy.core.fromnumeric import size

from setup_info import *

from data_loader import CustomCocoDetection, CustomCocoResult

def get_box_size_metric(gt):
    ## borrowed from coco: small, medium, large
    area_rngs = [ 0.0, 32 ** 2, 96 ** 2, 1e5 ** 2 ]
    ars = (gt[:, 2] - gt[:, 0])*(gt[:, 3] - gt[:, 1])
    ars = np.hstack(ars)
    hist = np.histogram(ars, area_rngs)[0]
    return hist

dataset = "imagenet_vid_argoverse_format"
data_loader = CustomCocoDetection(
    dataset_info[dataset]["train_root"], dataset_info[dataset]["train_json"], dataset
)

# size_metrics = []
# for i in range(0, len(data_loader)):
#     image, bboxes, classes = data_loader.__getitem__(i)
#     try:
#         metric = get_box_size_metric(bboxes.numpy())
#         size_metrics.append(metric)
#     except:
#         print(bboxes.numpy())

# print("area_size_metrics")
# print("mean", np.mean(size_metrics, axis=0), "std", np.std(size_metrics, axis=0))


def get_class_metric(gt):
    cls = [ 0 for x in range(0, 30) ]
    for g in gt:
        cls[g] += 1
    # print(gt)
    return cls

classes_info = {}
classes_metrics = []
for i in range(0, len(data_loader)):
    if i % 10 != 0:
        continue
    image, bboxes, classes = data_loader.__getitem__(i)

    try:
        metric = get_class_metric(classes.numpy())
        classes_metrics.append(metric)
    except:
        print(bboxes.numpy())

print("classes_metrics")
print("mean", np.mean(classes_metrics, axis=0))
print("std", np.std(classes_metrics, axis=0))
