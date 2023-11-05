import os
import copy
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.apis import init_detector

try:
    from scale_regressor import build_scale_regressor
    from transforms_utils import ImageTransformCPU, ImageTransformGPU
    from bbox_metrics import BboxMetrics, AdaptiveCrop
except:
    from .scale_regressor import build_scale_regressor
    from .transforms_utils import ImageTransformCPU, ImageTransformGPU
    from .bbox_metrics import BboxMetrics, AdaptiveCrop

np.set_printoptions(suppress=True)

class Tracktor(nn.Module):
    """
        A Barebones Implementation of Tracktor (without reID and camera motion compensation).
        Ref: Tracking without bells and whistles, https://arxiv.org/abs/1903.05625 [ICCV/CVPR 2019]
    """

    def __init__(
        self,
        two_stage_detector_name,
        dataset,
        two_stage_detector_config,
        two_stage_detector_checkpoint,
        device,
        regressor_config=None,
        preprocess_gpu=True,
        adaptive_crop=False,
        adaptive_crop_mode=None
    ):
        super().__init__()
        self.detector_name = two_stage_detector_name
        self.detector = init_detector(
            two_stage_detector_config, two_stage_detector_checkpoint, device
        )
        self.cfg = self.detector.cfg.copy()

        if preprocess_gpu:
            self.preprocessor = ImageTransformGPU(self.cfg.copy())
        else:
            self.preprocessor = ImageTransformCPU(self.cfg.copy())
        self.curr_scale = None

        self.bbox_metrics_calc = BboxMetrics(dataset)
        self.adaptive_crop = adaptive_crop
        if self.adaptive_crop:
            if adaptive_crop_mode in ["zoom", "discard"]:
                self.adaptive_crop_mode = adaptive_crop_mode
            else:
                raise Exception("Ill defined Adaptive Crop Mode!!")
            self.crop_extents = None
            self.crop_calc = AdaptiveCrop(dataset)

        self.device = device
        self.curr_scale_factor = torch.tensor([1, 1, 1, 1, 1]).to(device)

        self.scale_regressor = None
        if regressor_config is not None and "scale" in regressor_config:
            self.scale_regressor = build_scale_regressor(regressor_config["scale"])
            self.scale_regressor.load_state_dict(
                torch.load(regressor_config["scale"]["checkpoint_file"])
            )
            self.scale_regressor.to(device)

    def preprocess_image(self, img, crop_extents=None):
        if self.adaptive_crop:
            return self.preprocessor(img, self.device, crop_extents=crop_extents, adaptive_crop_mode=self.adaptive_crop_mode)
        return self.preprocessor(img, self.device)

    def change_scale(self, scale):
        self.curr_scale = scale
        self.preprocessor.change_scale(scale)

    def _change_scale_factor(self, scale_factor):
        self.curr_scale_factor = torch.tensor(scale_factor).to(self.device)
        self.curr_scale_factor = torch.cat(
            (self.curr_scale_factor, torch.tensor([1]).to(self.device))
        )

    def _result2detections(self, result):
        dets = []
        for i in range(len(result)):
            dets.append(result[i])
        if dets != []:
            dets = np.vstack(dets)
        else:
            dets = np.zeros((0, 5))
        dets = torch.from_numpy(dets).to(self.device)
        dets = dets * self.curr_scale_factor
        return [dets]

    def _decode_regressed_scale(self, regressed_scale, orig_shape):
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
        new_scale = int(new_scale)
        if new_scale > scales[0][1]:
            new_scale = scales[0][1]
        if new_scale < scales[-1][1]:
            new_scale = scales[-1][1]
        return new_scale
    
    def reset(self):
        pass
    
    def init(self, image, result):
        pass

    def forward(self, data, detections, get_metrics=False, im_shape=None):
        
        input_feat = self.detector.extract_feat(data["img"])
        if detections[0].size()[0] != 0:
            result = self.detector.roi_head.simple_test(
                input_feat, detections, data["img_metas"], rescale=True
            )[0]        
            if self.adaptive_crop:
                result = self.crop_calc.cropped_result_to_result(self.crop_extents, result)
        else:
            result = None
        if get_metrics:
            scales = self.regressor_config["scale"]["train_scales"]
            curr_scale = [ ( im_shape[1] - scales[-1][1] ) / ((scales[0][1] - scales[-1][1]) * 1.0) ]
            ada_scale = self.scale_regressor(input_feat).cpu().numpy()[0]
            class_info = self.bbox_metrics.get_class_info_metric(result)
            conf_info = self.bbox_metrics.get_confidence_metric(result)
            tb_lr_crop = self.adaptive_crop.get_crop_extents((im_shape[1], im_shape[0]), result)
            metrics = np.hstack([ curr_scale, ada_scale , class_info, conf_info, tb_lr_crop ])
            return result, metrics
        # if self.scale_regressor is not None:
        #     regressed_scale = self._decode_regressed_scale(
        #         self.scale_regressor(input_feat), data["img"].size()
        #     )
        #     return result, regressed_scale
        return result

    def track(self, image, prev_result, get_metrics=False):
        try:
            if self.adaptive_crop:
                # loc_spread = self.bbox_metrics_calc.get_localization_spread((image.shape[1], image.shape[0]), prev_result)
                self.crop_extents = self.crop_calc.get_crop_extents((image.shape[1], image.shape[0]), prev_result)
                data = self.preprocess_image(image, crop_extents=self.crop_extents)
            else:
                data = self.preprocess_image(image)
            self._change_scale_factor(data["img_metas"][0]["scale_factor"])
            # t = time.time()
            if self.adaptive_crop:
                prev_result = self.crop_calc.result_to_cropped_result(self.crop_extents, prev_result)
            # print( (time.time() - t)*1000 )
            detections = self._result2detections(prev_result)            
            result = self.forward(data, detections, get_metrics=get_metrics, im_shape=image.shape)
            if result is not None:
                return result
            return prev_result
        except:
            return prev_result


if __name__ == "__main__":
    pass