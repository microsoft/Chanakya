import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import glob
import pickle
import sys
import cv2
from PIL import  Image
import mmcv
import time
import pickle

models_name = ['yolo', 'fcos', 'faster_rcnn', 'faster_rcnn_101', 'cascade_rcnn']
type_data = sys.argv[1]

annType = 'bbox'
dataDir='../datasets/coco'
dataType= type_data + '2017'

annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
cocoGt=COCO(annFile)
YOUR_CAT_IDS = sorted(cocoGt.getCatIds())

out_file  =  "/mnt/IoU_results_coco/" + 'per_class_mAP_' + type_data + '.pkl'
final_result_dict = {}

for model_name in models_name:
    print("Here is the model name: ",model_name)
    resFile = "/mnt/IoU_results_coco/"+ model_name + "/" + type_data + "/" + type_data + ".json"
    cocoDt=cocoGt.loadRes(resFile)
    coco_eval = COCOeval(cocoGt, cocoDt, annType) 
    final_result_dict[model_name] = []
    stats = []
    for i in range(len(YOUR_CAT_IDS)):
        coco_eval.params.catIds = YOUR_CAT_IDS[i]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats.append(coco_eval.stats[1])

    final_result_dict[model_name] = stats

print(final_result_dict)
with open(out_file, 'wb') as handle:
    pickle.dump(final_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
