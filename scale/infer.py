import os
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import mmcv
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.apis import init_detector


from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

# def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
#     if isinstance(config, str):
#         config = mmcv.Config.fromfile(config)
#     elif not isinstance(config, mmcv.Config):
#         raise TypeError('config must be a filename or Config object, '
#                         f'but got {type(config)}')
#     if cfg_options is not None:
#         config.merge_from_dict(cfg_options)
#     config.model.pretrained = None
#     model = build_detector(config.model, test_cfg=config.test_cfg)
#     if checkpoint is not None:
#         map_loc = 'cpu' if device == 'cpu' else None
#         checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
#         if 'CLASSES' in checkpoint['meta']:
#             model.CLASSES = checkpoint['meta']['CLASSES']
#         else:
#             warnings.simplefilter('once')
#             warnings.warn('Class names are not saved in the checkpoint\'s '
#                           'meta data, use COCO classes by default.')
#             model.CLASSES = get_classes('coco')
#     model.cfg = config  # save the config in the model for convenience
#     model.to(device)
#     model.eval()
#     return model


def inference_detector(model, img, scale=None):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)

    # change scale as instructed
    if scale is not None:
        cfg.data.test.pipeline[1]["img_scale"] = scale

    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), "CPU inference with RoIPool is not supported currently."
        # just get the actual data from DataContainer
        data["img_metas"] = data["img_metas"][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result


# Specify the path to model config and checkpoint file

# config_file = 'saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

config_file = "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco.py"
checkpoint_file = "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco_20200603-ed16da04.pth"

# config_file = 'saved_models/base/yolov3/yolov3_d53_mstrain-608_273e_coco.py'
# checkpoint_file = 'saved_models/base/yolov3/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device="cuda:0")

# for x in model.modules():
#     print()
#     print()
# exit()

# test a single image and show the results
img = "image2.jpg"
img = mmcv.imread(img)

result = inference_detector(model, img)

# img = mmcv.imrescale(img, 0.1)
# t1 = time.time()
# iters = 1
# for i in range(0, iters):
# result = inference_detector(model, img)
# print(((time.time()-t1)*1000.0)/iters*1.0)

# model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)
