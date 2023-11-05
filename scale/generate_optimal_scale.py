import os
import pickle
import sys
import time
import copy

from numpy.lib.npyio import save

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import mmcv
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets.pipelines import Compose
from mmdet.apis import init_detector

from mmdet.core import get_classes, bbox2result, bbox2roi
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

from data_loader import CustomCocoDetection


def init_detector_train_style(
    config, checkpoint=None, device="cuda:0", cfg_options=None
):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    # config.model.train_cfg = None
    model = build_detector(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )
    if checkpoint is not None:
        map_loc = "cpu" if device == "cpu" else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "Class names are not saved in the checkpoint's "
                "meta data, use COCO classes by default."
            )
            model.CLASSES = get_classes("coco")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def infer_and_get_loss(model, model_type, img, gt_bboxes, gt_labels, scale=None):
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

    data["img_metas"] = data["img_metas"][0]
    data["gt_bboxes"] = [
        x * x.new_tensor(data["img_metas"][0]["scale_factor"]) for x in gt_bboxes
    ]
    data["gt_labels"] = gt_labels
    data["img"] = data["img"][0]

    with torch.no_grad():
        input_feat = model.extract_feat(data["img"])

        losses, results = [], []
        if model_type == "fcos":
            cls_scores, bbox_preds, centernesses = model.bbox_head(input_feat)
            losses = model.bbox_head.access_loss(
                cls_scores,
                bbox_preds,
                centernesses,
                data["gt_bboxes"],
                data["gt_labels"],
                data["img_metas"],
            )
            bbox_list = model.bbox_head.get_bboxes(
                cls_scores, bbox_preds, centernesses, data["img_metas"], rescale=True
            )
            results = [
                bbox2result(det_bboxes, det_labels, model.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
        if model_type == "faster_rcnn":

            proposal_list = model.rpn_head.simple_test_rpn(
                input_feat, data["img_metas"]
            )

            sampling_results = []
            for i in range(1):
                assign_result = model.roi_head.bbox_assigner.assign(
                    proposal_list[i], data["gt_bboxes"][i], None, data["gt_labels"][i]
                )
                sampling_result = model.roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    data["gt_bboxes"][i],
                    data["gt_labels"][i],
                    feats=[lvl_feat[i][None] for lvl_feat in input_feat],
                )
                sampling_results.append(sampling_result)

            rois = bbox2roi([res.bboxes for res in sampling_results])

            results = model.roi_head.simple_test(
                input_feat, proposal_list, data["img_metas"], rescale=True
            )

            bbox_results = model.roi_head._bbox_forward(input_feat, rois)

            (
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
            ) = model.roi_head.bbox_head.get_targets(
                sampling_results,
                data["gt_bboxes"],
                data["gt_labels"],
                model.train_cfg.rcnn,
            )
            losses = model.roi_head.bbox_head.access_loss(
                bbox_results["cls_score"],
                bbox_results["bbox_pred"],
                rois,
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
                data["img_metas"],
                reduction_override="none",
            )

    return input_feat, losses, results[0]


def draw_bboxes(img, bboxes, out_name, extra=None):
    if extra is None:
        extra = np.array([str(i + 1) for i in range(bboxes.shape[0])])
    mmcv.imshow_det_bboxes(img, bboxes, extra, show=False, out_file=out_name)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_foreground_bbox_indices(gt_bboxes, boxes_at_scale, threshold=0.5):
    index = np.zeros((len(boxes_at_scale), len(gt_bboxes)))
    for i in range(len(boxes_at_scale)):
        for j in range(len(gt_bboxes)):
            # print(boxes_at_scale[i, :],  gt_bboxes[j, :],  iou(boxes_at_scale[i, :], gt_bboxes[j, :]) )
            index[i, j] = iou(boxes_at_scale[i, :], gt_bboxes[j, :]) >= threshold
    index = np.sum(index, axis=1) > 0
    return index


def generate_optimal_scale_single(
    model, model_type, image, gt_bboxes, classes, meta, scales, device
):
    gt_bboxes_orig = gt_bboxes.clone().detach().cpu().numpy()
    if gt_bboxes_orig.shape[0] == 0:
        return None, None
    input_feats = []
    losses_at_diff_scales = []
    filter_indices_at_diff_scales = []
    for scale in scales:

        input_feat, loss_at_scale, result = infer_and_get_loss(
            model,
            model_type,
            image,
            [gt_bboxes.to(device)],
            [classes.to(device)],
            scale=scale,
        )
        if len(loss_at_scale["loss_cls"].shape) == 2:
            loss_at_scale["loss_cls"] = np.mean(
                loss_at_scale["loss_cls"], axis=1
            )  # bbox, class
        if len(loss_at_scale["loss_bbox"].shape) == 2:
            loss_at_scale["loss_bbox"] = np.mean(
                loss_at_scale["loss_bbox"], axis=1
            )  # bbox, 4
        loss_at_scale["loss"] = loss_at_scale["loss_cls"] + loss_at_scale["loss_bbox"]
        sort_index = np.argsort(loss_at_scale["loss"])[
            ::-1
        ]  # sort by descending, the paper is confusing but this one works
        for key in ["loss", "bboxes"]:
            loss_at_scale[key] = loss_at_scale[key][sort_index]
        input_feats.append(input_feat)
        filter_index = get_foreground_bbox_indices(
            gt_bboxes_orig, loss_at_scale["bboxes"]
        )
        filter_indices_at_diff_scales.append(filter_index)
        losses_at_diff_scales.append(loss_at_scale)

    min_len = np.min([np.sum(x == True) for x in filter_indices_at_diff_scales])
    optim_scale_idx = np.argmin(
        [
            np.sum(l["loss"][f][:min_len])
            for l, f in zip(losses_at_diff_scales, filter_indices_at_diff_scales)
        ]
    )
    optim_scale = scales[optim_scale_idx][1]
    # equation 3 in AdaScale
    optim_scale = ((optim_scale * 1.0) / meta["height"]) - (
        (scales[-1][1] * 1.0) / scales[0][1]
    )
    optim_scale = optim_scale / (
        (scales[0][1] * 1.0) / scales[-1][1] - (scales[-1][1] * 1.0) / scales[0][1]
    )
    optim_scale = 2 * optim_scale - 1

    return input_feats, optim_scale


def generate_optimal_scale(
    model, model_type, dataset, split, data_loader, scales, device
):
    # scales_hist = {
    #     x[1] : 0 for x in scales
    # }
    for i in range(1, len(data_loader), 5):
        image, gt_bboxes, classes = data_loader.__getitem__(i)
        meta = data_loader.__getmeta__(i)
        input_feats, optim_scale = generate_optimal_scale_single(
            model, model_type, image, gt_bboxes, classes, meta, scales, device
        )
        if optim_scale is None:
            continue
        save_path = "/datadrive/scale_data/{}/{}/{}_{}.pkl".format(
            model_type, split, dataset, meta["id"]
        )
        with open(save_path, "wb") as f:
            pickle.dump({"optim_scale": optim_scale, "input_feats": input_feats}, f)


if __name__ == "__main__":
    from setup_info import *

    # dataset = "coco"
    dataset = "imagenet_vid"
    model_type = "faster_rcnn"
    # model_type = "fcos"
    device_name = "cuda:0"

    split = "train"

    scales = [(2000, 600), (2000, 480), (2000, 360), (2000, 240)]  # (2000, 720),
    if model_type == "faster_rcnn":
        model = init_detector_train_style(
            models_info[dataset][model_type]["config_file"],
            models_info[dataset][model_type]["checkpoint_file"],
            device=device_name,
        )
    else:
        model = init_detector(
            models_info[dataset][model_type]["config_file"],
            models_info[dataset][model_type]["checkpoint_file"],
            device=device_name,
        )
    device = torch.device(device_name)
    data_loader = CustomCocoDetection(
        dataset_info[dataset]["{}_root".format(split)],
        dataset_info[dataset]["{}_json".format(split)],
        dataset,
    )

    # for i in range(5):
    #     image, bboxes, classes = data_loader.__getitem__(1)
    #     vals = infer_and_get_loss(model, model_type, image,
    #                         [ bboxes.to(device) ],
    #                         [ classes.to(device) ],
    #                         scale=(2000,600))

    generate_optimal_scale_single(
        model, model_type, dataset, split, data_loader, scales, device
    )
