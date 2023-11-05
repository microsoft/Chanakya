import os
import copy
import time

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.apis import init_detector

from mmdet.core import bbox2result

from scale_regressor import build_scale_regressor
from data_loader_harish import CustomCocoDetection, CustomCocoResult


class DetectorWithRegressors(nn.Module):
    def __init__(
        self,
        detector_name,
        detector_config,
        detector_checkpoint,
        device,
        backbone_type,
        regressor_config=None,
    ):
        super().__init__()
        self.detector_name = detector_name
        self.detector = init_detector(detector_config, detector_checkpoint, device)
        self.cfg = self.detector.cfg.copy()
        self.cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
        self.test_pipeline = Compose(self.cfg.data.test.pipeline)
        self.backbone_type = backbone_type
        self.num_proposals = None
        self.detector_category = {
            "single_stage": ["fcos", "yolov3"],
            "two_stage": ["faster_rcnn"],
        }
        self.regressor_config = regressor_config

        self.scale_regressor = None
        if regressor_config is not None and "scale" in regressor_config:
            self.scale_regressor = build_scale_regressor(regressor_config["scale"])
            self.scale_regressor.load_state_dict(torch.load(regressor_config["scale"]["checkpoint_file"]))
            self.scale_regressor.to(device)

    def preprocess_image(self, img):
        ## @TODO: Put all the pipeline stuff in the GPU
        ## Check sAP code on github (Added link on ADO)
        ## Paper claims we save 21 ms!!!!
        ## V.V.Imp!!
        device = next(self.detector.parameters()).device  # model device
        data = dict(img=img)
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(self.detector.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        else:
            for m in self.detector.modules():
                assert not isinstance(
                    m, RoIPool
                ), "CPU inference with RoIPool is not supported currently."
            # just get the actual data from DataContainer
            data["img_metas"] = data["img_metas"][0].data
        data["img_metas"] = data["img_metas"][0]
        data["img"] = data["img"][0]
        return data

    def change_scale(self, scale):
        self.cfg.data.test.pipeline[1]["img_scale"] = scale
        self.test_pipeline = Compose(self.cfg.data.test.pipeline)

    def change_num_proposals(self, num_proposals):
        self.num_proposals = num_proposals

    def unset_num_proposals(self):
        self.num_proposals = None

    def decode_regressed_scale(self, regressed_scale, orig_shape):
        orig_scale = list(orig_shape)[2] # N, C, H, W and we want H
        scales = self.regressor_config["scale"]["train_scales"]
        # inversion of eq (3) in AdaScale Paper
        new_scale = (regressed_scale + 1.0)*0.5*(
            (scales[0][1] * 1.0) / scales[-1][1] - (scales[-1][1] * 1.0) / scales[0][1]
        )
        new_scale = (new_scale + ((scales[-1][1] * 1.0) / scales[0][1]))*orig_scale
        new_scale = int(new_scale)
        if new_scale > scales[0][1]:
            new_scale = scales[0][1]
        if new_scale < scales[-1][1]:
            new_scale = scales[-1][1]
        return new_scale

    def forward(self, data):
        input_feat = self.detector.extract_feat(data["img"])
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
        if self.scale_regressor is not None:
            regressed_scale = self.decode_regressed_scale(self.scale_regressor(input_feat), data['img'].size())
            return bbox_results[0], regressed_scale
        return bbox_results[0]

    def detect(self, image):
        data = self.preprocess_image(image)
        result = self.forward(data)
        return result


if __name__ == "__main__":
    import random
    import numpy as np
    import itertools

    from setup_info import *

    model_name = "faster_rcnn"
    dataset = "coco"
    device = "cuda:1"
    data_loader = CustomCocoDetection(
        dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset
    )
    det = DetectorWithRegressors(
        model_name,
        models_info[dataset][model_name]["config_file"],
        models_info[dataset][model_name]["checkpoint_file"],
        device,
        "resnet_50",
        None #models_info[dataset][model_name]["regressors"],
    )

    proposal_vals = [100, 300, 500, 1000]
    scale_vals = [(2000, 600), (2000, 480), (2000, 360), (2000, 240)]

    for pr, scale in itertools.product(proposal_vals, scale_vals):
        det.change_num_proposals(pr)
        det.change_scale(scale)

        results_obj = CustomCocoResult(dataset, dataset_info[dataset]["test_json"])
        
        times = []
        for i in range(500):
            image, bboxes, classes = data_loader.__getitem__(i)
            meta = data_loader.__getmeta__(i)
            data = det.preprocess_image(image)
            ts = time.time()
            result = det(data)
            times.append((time.time() - ts) * 1000)
            results_obj.add_mmdet_results(meta, result)
        print(pr, scale, np.mean(times[1:]))
        results_obj.evaluate()

    # det.detector.show_result(copy.deepcopy(image), result[0], out_file="out_class.jpg")
