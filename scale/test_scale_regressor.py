import os
import random

import torch

from scale_regressor import build_scale_regressor
from data_loader import CustomCocoDetection

from generate_optimal_scale import (
    generate_optimal_scale_single,
    init_detector_train_style,
    init_detector,
)

from setup_info import *
from sklearn.metrics import r2_score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(
    model,
    model_type,
    regressor,
    data_loader,
    scales,
    device,
):
    iter_arr = list(range(0, len(data_loader), 30))
    random.shuffle(iter_arr)
    num = 0
    gt = []
    pred = []
    for i in iter_arr:
        num += 1

        image, gt_bboxes, classes = data_loader.__getitem__(i)
        meta = data_loader.__getmeta__(i)
        input_feats, optim_scale = generate_optimal_scale_single(
            model, model_type, image, gt_bboxes, classes, meta, scales, device
        )
        if optim_scale is None:
            continue
        gt.append(optim_scale)
        optim_scale = torch.tensor(optim_scale).to(device)
        feats = random.choice(input_feats)
        scale_pred = regressor(feats)
        pred.append(scale_pred.cpu().numpy()[0][0])
        # print(i)
    print(len(gt))
    print(r2_score(gt, pred))

def test_loop():


    # dataset = "argoverse_coco_finetune"
    dataset = "imagenet_vid"
    train_data_loader = CustomCocoDetection(
        dataset_info[dataset]["test_root"],
        dataset_info[dataset]["test_json"],
        dataset,
    )

    model_type = "faster_rcnn"
    # model_type = "fcos"
    device_name = "cuda:0"

    model_info = models_info[dataset][model_type]
    if model_type == "faster_rcnn":
        model = init_detector_train_style(
            model_info["config_file"],
            model_info["checkpoint_file"],
            device=device_name,
        )
    else:
        model = init_detector(
            model_info["config_file"],
            model_info["checkpoint_file"],
            device=device_name,
        )
    device = torch.device(device_name)

    scale_reg_info = model_info["regressors"]["scale"]
    regressor = build_scale_regressor(scale_reg_info)
    regressor.to(device)
    regressor_checkpoint_path = scale_reg_info["checkpoint_file"]

    scales = scale_reg_info["train_scales"]

    regressor.load_state_dict(torch.load(regressor_checkpoint_path))

    with torch.no_grad():
        test(
            model,
            model_type,
            regressor,
            train_data_loader,
            scales,
            device,
        )
        

if __name__ == "__main__":
    test_loop()
