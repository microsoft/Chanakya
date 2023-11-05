import os
import copy
import time
from matplotlib.colors import Normalize

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.apis import init_detector

from mmdet.core import bbox2result

try:    
    from scale_regressor import build_scale_regressor
    from complexity_regressor import build_complexity_regressor
    from data_loader import CustomCocoDetection, CustomCocoResult
    from transforms_utils import ImageTransformCPU, ImageTransformGPU
    from bbox_metrics import BboxMetrics, AdaptiveCrop
except:
    from .scale_regressor import build_scale_regressor
    from .complexity_regressor import build_complexity_regressor
    from .data_loader import CustomCocoDetection, CustomCocoResult
    from .transforms_utils import ImageTransformCPU, ImageTransformGPU
    from .bbox_metrics import BboxMetrics, AdaptiveCrop

np.set_printoptions(precision=3)


class DetectorWithRegressors(nn.Module):
    def __init__(
        self,
        detector_name,
        dataset,
        detector_config,
        detector_checkpoint,
        device,
        regressor_config=None,
        preprocess_gpu=True,
    ):
        super().__init__()
        self.dataset = dataset
        self.detector_name = detector_name
        self.detector = init_detector(detector_config, detector_checkpoint, device)
        self.num_proposals = None
        self.detector_category = {
            "single_stage": ["fcos", "yolov3"],
            "two_stage": ["faster_rcnn", "cascade_rcnn", "faster_rcnn_101", "swin_tiny_mask_rcnn", "swin_tiny_cascade_rcnn"],
        }
        self.device = device

        if preprocess_gpu == True:
            self.preprocessor = ImageTransformGPU(self.detector.cfg.copy())
        else:
            self.preprocessor = ImageTransformCPU(self.detector.cfg.copy())
        self.curr_scale = self.detector.cfg.data.test.pipeline[1]["img_scale"]

        self.bbox_metrics = BboxMetrics(self.dataset)
        self.adaptive_crop = AdaptiveCrop(self.dataset)
        self.regressor_config = regressor_config
        self.scale_regressor = None
        if regressor_config is not None and "scale" in regressor_config:
            self.scale_regressor = build_scale_regressor(regressor_config["scale"])
            self.scale_regressor.load_state_dict(
                torch.load(regressor_config["scale"]["checkpoint_file"])
            )
            self.scale_regressor.to(device)
        if regressor_config is not None and "complexity" in regressor_config:
            self.complexity_regressor = build_complexity_regressor(regressor_config["complexity"])
            self.complexity_regressor.load_state_dict(
                torch.load(regressor_config["complexity"]["checkpoint_file"], map_location=device)
            )
            self.complexity_regressor.to(device)

    def preprocess_image(self, img):
        return self.preprocessor(img, self.device)

    def change_scale(self, scale):
        self.curr_scale = scale
        self.preprocessor.change_scale(scale)

    def change_num_proposals(self, num_proposals):
        self.num_proposals = num_proposals

    def unset_num_proposals(self):
        self.num_proposals = None

    def decode_regressed_scale(self, regressed_scale, orig_shape):
        orig_scale = list(orig_shape)[2]  # N, C, H, W and we want H
        scales = self.regressor_config["scale"]["train_scales"]
        # inversion of eq (3) in AdaScale Paper
        new_scale = (
            (regressed_scale + 1.0)
            * 0.5
            * (
                (scales[0][1] * 1.0) / scales[-1][1]
                - (scales[-1][1] * 1.0) / scales[0][1]
            )
        )
        new_scale = (new_scale + ((scales[-1][1] * 1.0) / scales[0][1])) * orig_scale
        # round to int
        new_scale = int(new_scale)
        # clipping
        if new_scale > scales[0][1]:
            new_scale = scales[0][1]
        if new_scale < scales[-1][1]:
            new_scale = scales[-1][1]
        return new_scale

    def forward(self, data, get_metrics=False, im_shape=None, predict_scale=False, get_switch_metric=False, get_area_metric=False):
        input_feat = self.detector.extract_feat(data["img"])
        # for i in range(len(input_feat)):
        #     print(input_feat[i].size())
        # print(len(input_feat))
        if self.detector_name in self.detector_category["single_stage"]:
            outs = self.detector.bbox_head(input_feat)
            bbox_list = self.detector.bbox_head.get_bboxes(
                *outs, data["img_metas"], rescale=True
            )
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.detector.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
        elif self.detector_name in self.detector_category["two_stage"]:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                input_feat, data["img_metas"]
            )
            if self.num_proposals is not None:
                proposal_list = [p[: self.num_proposals, :] for p in proposal_list]
            bbox_results = self.detector.roi_head.simple_test(
                input_feat, proposal_list, data["img_metas"], rescale=True
            )
        if predict_scale:
            ada_scale = self.decode_regressed_scale(self.scale_regressor(input_feat), data['img'].size())
            return bbox_results[0], ada_scale

        if get_metrics:
            scales = self.regressor_config["scale"]["train_scales"]
            curr_scale = [ ( self.curr_scale[1] - scales[-1][1] ) / ((scales[0][1] - scales[-1][1]) * 1.0) ] # 1 * 1
            ada_scale = self.decode_regressed_scale(self.scale_regressor(input_feat), data['img'].size())
            ada_scale = [ ( ada_scale - scales[-1][1] ) / ((scales[0][1] - scales[-1][1]) * 1.0) ] # 1 * 1
            class_info = self.bbox_metrics.get_class_info_metric(bbox_results[0], normalized=True) # 8 * 1
            conf_info = self.bbox_metrics.get_confidence_metric(bbox_results[0]) # 2 * 1
            tb_lr_crop = self.adaptive_crop.get_crop_extents((im_shape[1], im_shape[0]), bbox_results[0], normalized=True) # 3 * 1
                        
            if get_switch_metric: # 3 * 1
                switch_temp = self.complexity_regressor(input_feat)
                switch_score = torch.softmax(switch_temp, dim = 1).cpu().numpy()[0]
                metrics = np.hstack([ curr_scale, ada_scale , class_info, conf_info, tb_lr_crop, switch_score ])
            else:
                metrics = np.hstack([ curr_scale, ada_scale , class_info, conf_info, tb_lr_crop ])
            
            if get_area_metric: # 3 * 1
                area_info = self.bbox_metrics.get_box_size_metric(bbox_results[0], normalized=True)
                metrics = np.hstack([metrics, area_info])

            return bbox_results[0], metrics
        return bbox_results[0]

    def parse_result_for_sap(self, result):
        bboxes = []
        labels = []
        if self.detector_name == "swin_tiny_mask_rcnn" or self.detector_name == "swin_tiny_cascade_rcnn":
            result = result[0]
        for i in range(len(result)):
            if len(result[i].shape) > 1:
                for bbox in result[i]:
                    bboxes.append(bbox.copy())
                    labels.append(i)
            else:
                bboxes.append(result[i].reshape((1,5)).copy())
                labels.append(i)
        if bboxes != []:
            return np.vstack(bboxes), np.array(labels)
        else:
            return np.zeros((0,5)), np.zeros((0,)).astype(int)

    def detect(self, image, get_metrics=False, predict_scale=False, get_switch_metric=False, get_area_metric=False):
        with torch.no_grad():
            data = self.preprocess_image(image)
            result = self.forward(data, get_metrics=get_metrics, im_shape=image.shape, predict_scale=predict_scale, get_switch_metric=get_switch_metric, get_area_metric=get_area_metric)
            return result


if __name__ == "__main__":
    import random
    import numpy as np
    import itertools

    from setup_info import *

    model_name = "faster_rcnn"
    dataset = "argoverse"
    device = "cuda:0"
    data_loader = CustomCocoDetection(
        dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
    )
    det = DetectorWithRegressors(
        model_name,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        device,
        # models_info[dataset][model_name]["regressors"],
    )
    # exit()

    # proposal_vals = [100, 300, 500, 1000]
    # scale_vals = [(2000, 600), (2000, 480), (2000, 360), (2000, 240)]

    proposal_vals = [300]
    scale_vals = [(2000,  640), (2000, 480), (2000, 360)]

    for pr, scale in itertools.product(proposal_vals, scale_vals):
        det.change_num_proposals(pr)
        det.change_scale(scale)

        results_obj = CustomCocoResult(dataset, dataset_info[dataset]["test_json"])

        times = []
        for i in range(2000):
            image, bboxes, classes = data_loader.__getitem__(i)
            meta = data_loader.__getmeta__(i)
            ts = time.time()
            # data = det.preprocess_image(image)
            # result = det(data)
            result = det.detect(image)#, get_metrics=True)
            _ = det.parse_result_for_sap(result)
            times.append((time.time() - ts) * 1000)
            results_obj.add_mmdet_results(meta, result)
            break
        break
        print(pr, scale, np.mean(times[1:]))
        results_obj.evaluate()
    # det.detector.show_result(copy.deepcopy(image), result[0], out_file="out_class.jpg")
