import pandas as pd
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


file_path_model = "/mnt/IoU_results/" 
file_path = "/mnt/IoU_results/test_yolov3_fcos_faster_rcnn_imagenet_vid_sd_sqID_tag.csv"
columns_name = ['image_id', 'total_mean']
columns_name_changed = ['seq_id','image_id', 'SD', 'image_tag']

data = pd.read_csv(file_path, usecols=columns_name_changed)

# models_name = ['yolov3', 'faster_rcnn', 'fcos']
models_name = ['yolov3']
easy_sequences = [7020, 7022, 7024, 7026, 8000, 8002, 9000, 9001, 15000, 16000, 16013, 18000, 18001, 19002, 23012, 23013, 26001, 26002, 29002, 30000, 35000]
hard_sequences = [122000, 77000, 4, 128001, 99001, 31001, 33002, 16007, 17001, 105000, 165000, 118001, 129000, 81000, 0, 171000, 2, 118009, 161001, 13001, 33001]
# easy_sequences = [0]
# hard_sequences = [1]
sq_values  =  {}
k = int(len(data))
print(k)
for i in range(k):
    sq_values[int(data['seq_id'][i])] = []
for i in range(k):
    sq_values[int(data['seq_id'][i])].append(int(data['image_id'][i]))

# for i in range(len(easy_sequences)):
#     image_id_list = sq_values[easy_sequences[i]]
#     plt.figure()
#     print(image_id_list)


annType = 'bbox'
dataDir='/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-argoverse-format'
type_data = sys.argv[1]


mode = sys.argv[2]
annFile = '/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-argoverse-format/annotations/instances_val.json'
cocoGt=COCO(annFile)


def draw_bboxes_dt(img, bboxes, out_name):
    # print(bboxes)
    mmcv.imshow_det_bboxes(img, bboxes, 
        np.array([ str(i + 1) for i in range(bboxes.shape[0]) ]),
        show=False, out_file=out_name)

out_image_path = '/mnt/IoU_results/video_seq/'

for model in models_name:
    print("Model name :", model)
    resFile = "/mnt/IoU_results/"+ model + "/" + type_data + "/" + type_data + ".json"
    cocoDt=cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval._prepare() 
    
    for j in range(len(easy_sequences)):
        count = 0
        print("Easy Seq No:", easy_sequences[j])
        bbox_imgIDs = sq_values[easy_sequences[j]]
        for imgId in bbox_imgIDs:

            if mode == "ground_truth":
                gts = cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=imgId))
                bbox = []
                for i in range(len(gts)):
                    [x,y,w,h] = gts[i]['bbox']
                    changed_bbox = [int(x), int(y), int(x+w), int(y+h)]
                    bbox.append(changed_bbox)
            else:
                dts = cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=imgId))    
                bbox = []
                for i in range(len(dts)):
                    [x,y,w,h] = dts[i]['bbox']
                    # print(dts[i]['bbox'])
                    changed_bbox = [int(x), int(y), int(x+w), int(y+h)]
                    bbox.append(changed_bbox)

            bbox = np.array(bbox)
            img = cocoGt.loadImgs(imgId)
            # print(img[0]['file_name'])
            dataType = "val"
            img = mmcv.imread('%s/%s/%s'%(dataDir,dataType,img[0]['file_name']))
            if mode=="ground_truth":
                if count == 0:
                    directory = str(easy_sequences[j])
                    path = os.path.join(out_image_path + str(mode) + '/easy', directory)
                    try:
                        os.mkdir(path)
                    except:
                        pass  
                out_file_name = out_image_path + str(mode) + '/' + 'easy' + '/' +str(easy_sequences[j]) + '/' + str(mode) + '_' + str(imgId)+ '.jpg'
                count += 1
            else:
                if count == 0:
                    directory = str(easy_sequences[j])
                    path = os.path.join(out_image_path + str(model) + '/easy', directory)
                    try:
                        os.mkdir(path)
                    except:
                        pass
                out_file_name = out_image_path + str(model) + '/' + 'easy' + '/' +str(easy_sequences[j]) + '/' +  str(mode) + '_' + str(model) + '_' + str(imgId)+ '.jpg'
                count +=1
            if bbox==[]:
                continue
            try:
                draw_bboxes_dt(img, bbox, out_file_name)
            except:
                pass


    for j in range(len(hard_sequences)):
        count = 0
        print("Hard Seq No:", hard_sequences[j])
        bbox_imgIDs = sq_values[hard_sequences[j]]
        for imgId in bbox_imgIDs:

            if mode == "ground_truth":
                gts = cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=imgId))
                bbox = []
                for i in range(len(gts)):
                    [x,y,w,h] = gts[i]['bbox']
                    changed_bbox = [int(x), int(y), int(x+w), int(y+h)]
                    bbox.append(changed_bbox)
            else:
                dts = cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=imgId))    
                bbox = []
                for i in range(len(dts)):
                    [x,y,w,h] = dts[i]['bbox']
                    # print(dts[i]['bbox'])
                    changed_bbox = [int(x), int(y), int(x+w), int(y+h)]
                    bbox.append(changed_bbox)

            bbox = np.array(bbox)
            img = cocoGt.loadImgs(imgId)
            # print(img[0]['file_name'])
            dataType = "val"
            img = mmcv.imread('%s/%s/%s'%(dataDir,dataType,img[0]['file_name']))
            if mode=="ground_truth":
                if count == 0:
                    directory = str(hard_sequences[j])
                    path = os.path.join(out_image_path + str(mode) + '/hard', directory)
                    try:
                        os.mkdir(path)
                    except:
                        pass  
                out_file_name = out_image_path + str(mode) + '/' + 'hard' + '/' + str(hard_sequences[j]) + '/' + str(mode) + '_' + str(imgId)+ '.jpg'
                count += 1
            else:
                if count == 0:
                    directory = str(hard_sequences[j])
                    path = os.path.join(out_image_path + str(model) + '/hard', directory)
                    try:
                        os.mkdir(path)
                    except:
                        pass    
                out_file_name = out_image_path + str(model) + '/' + 'hard' + '/' + str(hard_sequences[j]) + '/' +  str(mode) + '_' + str(model) + '_' + str(imgId)+ '.jpg'
                count +=1
            if bbox==[]:
                continue
            try:
                draw_bboxes_dt(img, bbox, out_file_name)
            except:
                pass
