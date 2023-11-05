import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval_custom_2 import COCOeval
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

model_name = sys.argv[1]
# model_name = ['yolov3', 'fcos', 'faster_rcnn']

#  sd = [0.00414720278362392, 0.0261550180157663, 0.126649385355781, 0.218882770207344, 0.343756504141631 ,0.455149699111682] -> Train
# -> 
annType = 'bbox'
dataDir='/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/'
type_data = sys.argv[2]
dataType= type_data + '2017'
IoU_file_save = '/mnt/IoU_results_argoverse/' + model_name + '/' + type_data + '/'
# "test_json": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/annotations/instances_val.json",

# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
if type_data == "test":
    annFile = '/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/annotations/val_full.json'
elif type_data  == "train":
    annFile = '/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/annotations/train_full.json'

cocoGt=COCO(annFile)

resFile = "/mnt/IoU_results_argoverse/"+ model_name + "/" + type_data + "/" + type_data + "_full.json"
cocoDt=cocoGt.loadRes(resFile)
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval._prepare() 
cocoEval.evaluate(IoU_file_save, type_data)

# cocoEval.accumulate()
# print(cocoEval.eval['scores'].shape)    (10, 101, 80, 4, 3)

# print(cocoEval.ious)
# for i in range(len(cocoEval.evalImgs)):
#     # print("Hi")
#     if cocoEval.evalImgs[i] is not None:
#         print(cocoEval.evalImgs[i]['aRng'])
#     else:
#         # print("None")
