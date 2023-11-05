import warnings
import numpy as np


import mmcv
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose

import torch
import torch.nn.functional as F

# Adapted from:
# https://github.com/mtli/sAP/blob/master/det/det_apis.py
class ImageTransformGPU(object):
    """Preprocess an image ON A GPU.
    """

    def __init__(self, detector_config):
        self.cfg = detector_config
        self.size_divisor = None
        self.curr_scale = self.cfg.data.test.pipeline[1]["img_scale"]
        self.keep_ratio = True
        for tf in self.cfg.data.test.pipeline[1]["transforms"]:
            if tf["type"] == "Resize":
                self.keep_ratio = tf["keep_ratio"]
            elif tf["type"] == "Normalize":
                self.mean = torch.tensor(tf["mean"], dtype=torch.float32)
                self.mean_np = np.array(tf["mean"])
                self.std = torch.tensor(tf["std"], dtype=torch.float32)
                self.std_np = np.array(tf["std"])
                self.std_inv = 1 / self.std
                self.to_rgb = (tf["to_rgb"],)  # assuming already in RGB
            elif tf["type"] == "Pad":
                self.size_divisor = tf["size_divisor"]

    def _get_new_size(self, w, h, scale, keep_ratio):
        if keep_ratio:
            if isinstance(scale, (float, int)):
                if scale <= 0:
                    raise ValueError(
                        "Invalid scale {}, must be positive.".format(scale)
                    )
                scale_factor = scale
            elif isinstance(scale, tuple):
                max_long_edge = max(scale)
                max_short_edge = min(scale)
                scale_factor = min(
                    max_long_edge / max(h, w), max_short_edge / min(h, w)
                )
            else:
                raise TypeError(
                    "Scale must be a number or tuple of int, but got {}".format(
                        type(scale)
                    )
                )

            new_size = (round(h * scale_factor), round(w * scale_factor))
        else:
            new_size = scale
            w_scale = new_size[1] / w
            h_scale = new_size[0] / h
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )
        return new_size, scale_factor

    def change_scale(self, scale, keep_ratio=True):
        self.curr_scale = scale
        self.keep_ratio = keep_ratio

    def preprocess(self, img, device, crop_extents=None, adaptive_crop_mode=None):
        ori_img_shape = img.shape
        if crop_extents != None and adaptive_crop_mode == "zoom":
            top, bottom, left, right = crop_extents
            img = img[top:bottom, left:right]
            ori_img_shape = img.shape

        h, w = img.shape[:2]
        new_size, scale_factor = self._get_new_size(
            w, h, self.curr_scale, self.keep_ratio
        )
        img_shape = (*new_size, 3)

        img = torch.from_numpy(img).to(device).float()
        # to BxCxHxW
        img = img.permute(2, 0, 1).unsqueeze_(0)

        if new_size[0] != img.shape[1] or new_size[1] != img.shape[2]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore the align_corner warnings
                img = F.interpolate(img, new_size, mode="bilinear")
        for c in range(3):
            img[:, c, :, :].sub_(self.mean[c]).mul_(self.std_inv[c])

        if crop_extents != None and adaptive_crop_mode == "discard":
            np_scale_factor = np.array([scale_factor for i in range(4)]).astype(np.float32)
            top, bottom, left, right = (np.array(crop_extents) * np_scale_factor).astype(int)
            img = img[:, :, top:bottom, left:right] # it's a torch tensor now
            # ori_img_shape = tuple(img.shape[2:])

        if self.size_divisor is not None:
            pad_h = (
                int(np.ceil(new_size[0] / self.size_divisor)) * self.size_divisor
                - new_size[0]
            )
            pad_w = (
                int(np.ceil(new_size[1] / self.size_divisor)) * self.size_divisor
                - new_size[1]
            )
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
            pad_shape = (img.shape[2], img.shape[3], 3)
        else:
            pad_shape = img_shape
        return img, ori_img_shape, img_shape, pad_shape, scale_factor

    def __call__(self, img, device, crop_extents=None, adaptive_crop_mode=None):
        img, ori_img_shape, img_shape, pad_shape, scale_factor = self.preprocess(img, device, crop_extents=crop_extents, adaptive_crop_mode=adaptive_crop_mode)
        # for update in bbox_head.py
        if type(scale_factor) is int:
            scale_factor = float(scale_factor)
        scale_factor = np.array([scale_factor for i in range(4)]).astype(np.float32)
        img_meta = [
            dict(
                filename=None,
                ori_filename=None,
                ori_shape=ori_img_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=False,
                img_norm_cfg=dict(
                    mean=self.mean_np, std=self.std_np, to_rgb=self.to_rgb
                ),
            )
        ]
        return dict(img=img, img_metas=img_meta)


class ImageTransformCPU(object):
    def __init__(self, detector_config):
        self.cfg = detector_config  # self.detector.cfg.copy()
        self.cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
        self.test_pipeline = Compose(self.cfg.data.test.pipeline)

    def change_scale(self, scale, keep_ratio=True):
        self.cfg.data.test.pipeline[1]["img_scale"] = scale
        for idx, tf in enumerate(self.cfg.data.test.pipeline[1]["transforms"]):
            if tf["type"] == "Resize":
                break
        self.cfg.data.test.pipeline[1]["transforms"][idx]["keep_ratio"] = keep_ratio
        self.test_pipeline = Compose(self.cfg.data.test.pipeline)

    def __call__(self, img, device, crop_extents=None):
        # TODO: Add crop extents usage!!!
        data = dict(img=img)
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if device is not None:
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


if __name__ == "__main__":
    pass
