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

def train(
    model,
    model_type,
    regressor,
    data_loader,
    scales,
    device,
    epoch,
    criterion,
    optimizer,
):
    loss_avg = AverageMeter()
    iter_arr = list(range(0, len(data_loader)))
    random.shuffle(iter_arr)
    num = 0
    for i in iter_arr:
        num += 1
        optimizer.zero_grad()

        image, gt_bboxes, classes = data_loader.__getitem__(i)
        meta = data_loader.__getmeta__(i)
        input_feats, optim_scale = generate_optimal_scale_single(
            model, model_type, image, gt_bboxes, classes, meta, scales, device
        )
        if optim_scale is None:
            continue
        optim_scale = torch.tensor(optim_scale).to(device)
        feats = random.choice(input_feats)
        scale_pred = regressor(feats)
        loss = criterion(scale_pred, optim_scale)
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())
        if num % 500 == 0:
            print(
                "{}: {} / {} {:.3f}".format(epoch, num, len(data_loader), loss_avg.avg)
            )


def test():
    pass


def train_test_loop():


    dataset = "argoverse_coco_finetune"
    # dataset = "imagenet_vid"
    train_data_loader = CustomCocoDetection(
        dataset_info[dataset]["train_root"],
        dataset_info[dataset]["train_json"],
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
    epochs = 2

    if os.path.exists(regressor_checkpoint_path):
        regressor.load_state_dict(torch.load(regressor_checkpoint_path))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.0001)
    for epoch in range(epochs):
        train(
            model,
            model_type,
            regressor,
            train_data_loader,
            scales,
            device,
            epoch,
            criterion,
            optimizer,
        )
        torch.save(regressor.state_dict(), regressor_checkpoint_path)


if __name__ == "__main__":
    train_test_loop()
