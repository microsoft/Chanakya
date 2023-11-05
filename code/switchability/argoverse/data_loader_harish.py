import os
import os.path
import json
from typing import Any, Callable, Optional, Tuple, List

from numpy.lib.npyio import save

import torch
import torch.utils.data as data
import mmcv
import numpy as np

# ARGOVERSE_CLASSES = (
#     'person', #0
#     'bicycle', #1
#     'car', #2
#     'motorcycle', #3 
#     'bus', #4
#     'truck', #5
#     'traffic_light', #6 
#     'stop_sign' #7
# )

COCO_CLASSES = (
    "person", #0
    "bicycle", #1
    "car", #2
    "motorcycle", #3
    "airplane", #4
    "bus", #5
    "train", #6
    "truck", #7
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

IMAGENET_VID_CLASSES = (
    "n01503061",
    "n01662784",
    "n01674464",
    "n01726692",
    "n02062744",
    "n02084071",
    "n02118333",
    "n02121808",
    "n02129165",
    "n02129604",
    "n02131653",
    "n02324045",
    "n02342885",
    "n02355227",
    "n02374451",
    "n02391049",
    "n02402425",
    "n02411705",
    "n02419796",
    "n02484322",
    "n02503517",
    "n02509815",
    "n02510455",
    "n02691156",
    "n02834778",
    "n02924116",
    "n02958343",
    "n03790512",
    "n04468005",
    "n04530566",
)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CustomCocoDetection(data.Dataset):
    def __init__(
        self, root: str, annFile: str, dataset_name: str
    ):
        from pycocotools.coco import COCO

        self.root = root
        with open(annFile) as f:
            self.coco_json = json.load(f)
        self.coco = COCO(annFile)
        self.dataset_name = dataset_name
        if dataset_name == "coco" or dataset_name == "argoverse_coco_finetune" or dataset_name == "argoverse":
            self.cat_ids = self.coco.get_cat_ids(cat_names=COCO_CLASSES)
        elif (
            dataset_name == "imagenet_vid"
            or dataset_name == "imagenet_vid_argoverse_format"
        ):
            self.cat_ids = self.coco.get_cat_ids(cat_names=IMAGENET_VID_CLASSES)
        else:
            raise Exception("Bad dataset name!!!")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        if dataset_name == "argoverse":
            self.inverse_mapping = {}
            for coco_idx, argo_idx in enumerate(self.coco_json["coco_mapping"]):
                self.inverse_mapping[argo_idx] = coco_idx + 1

    def _load_image(self, id: int):
        im_info = self.coco.loadImgs(id)[0]
        if self.dataset_name == "argoverse":
            path = im_info["name"]
            path = os.path.join(self.coco_json["seq_dirs"][im_info["sid"]], path)
        else:
            path = im_info["file_name"]
        return mmcv.imread(os.path.join(self.root, path))

    def _load_target(self, id, H, W) -> List[Any]:
        target = self.coco.loadAnns(self.coco.getAnnIds(id))
        bboxes = []
        classes = []
        for x in target:
            x1, y1, w, h = x["bbox"]
            inter_w = max(0, min(x1 + w, W) - max(x1, 0))
            inter_h = max(0, min(y1 + h, H) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if x["area"] <= 0 or w < 1 or h < 1:
                continue
            if x["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            bboxes.append(bbox)
            if self.dataset_name == "argoverse":
                cat_id = x["category_id"]
                classes.append(self.inverse_mapping[cat_id]) # send back the coco cat id
            else:
                classes.append(self.cat2label[x["category_id"]])
        bboxes = torch.Tensor(np.array(bboxes))
        classes = torch.LongTensor(classes)
        return bboxes, classes

    def get_sequences_list(self):
        seqs_list = np.unique([img["sid"] for img in self.coco.imgs.values()])
        return seqs_list

    def get_sequence(self, seq_id: int):
        frame_list = [img for img in self.coco.imgs.values() if img["sid"] == seq_id]
        return frame_list

    def get_frame(self, id: int):
        image = self._load_image(id)
        H, W, C = image.shape
        bboxes, classes = self._load_target(id, H, W)
        return image, bboxes, classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        H, W, C = image.shape
        bboxes, classes = self._load_target(id, H, W)
        return image, bboxes, classes

    def __getmeta__(self, index: int, is_image_id: bool = False):
        if is_image_id:
            return self.coco.loadImgs(index)[0]
        return self.coco.loadImgs(self.ids[index])[0]

    def __len__(self) -> int:
        return len(self.ids)


class CustomCocoResult:
    def __init__(self, dataset_name, gt_json):
        from pycocotools.coco import COCO

        self.results = []
        self.imgIds = []

        self.dataset = dataset_name
        self.coco = COCO(gt_json)
        with open(gt_json) as f:
            self.coco_json = json.load(f)


        if dataset_name == "coco" or dataset_name == "argoverse_coco_finetune" or dataset_name == "argoverse":
            self.cat_ids = self.coco.get_cat_ids(cat_names=COCO_CLASSES)
        elif (
            dataset_name == "imagenet_vid"
            or dataset_name == "imagenet_vid_argoverse_format"
        ):
            self.cat_ids = self.coco.get_cat_ids(cat_names=IMAGENET_VID_CLASSES)
        else:
            raise Exception("Bad dataset name!!!")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}

    def add_mmdet_results(self, im_info, result, seq):
        self.imgIds.append(im_info["id"])
        if result is None:
            return
        for i in range(len(result)):
            label = self.label2cat[i]
            if self.dataset == "argoverse":
                # convert coco "mmdetection pred label" (not cat id, ref self.cat2label) to argoverse cat id
                label = self.coco_json["coco_mapping"][i]  
            for det in result[i]:
                try:
                    x1, y1, x2, y2, s = det
                except TypeError:
                    print(result[i], det)
                    raise Exception("Stop!")
                w = x2 - x1
                h = y2 - y1
                res_dict = {
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(s),
                    "category_id": label,
                    "image_id": im_info["id"],
                    "seq_id" : int(seq)
                }
                self.results.append(res_dict)

    def save_result_json(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.results, f, cls=NumpyEncoder)

    def evaluate(self, save_as=None, cat_ids=None):
        # print(self.imgIds)
        # print(np.unique([ x["image_id"] for x in self.results]))
        from pycocotools.cocoeval import COCOeval

        if save_as is None:
            save_as = "tmp.json"
        self.save_result_json(save_as)
        coco_dt = self.coco.loadRes(save_as)
        coco_eval = COCOeval(self.coco, coco_dt, "bbox")
        coco_eval.params.imgIds = self.imgIds
        if cat_ids is not None:
            coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        if save_as is None:
            os.remove("tmp.json")


if __name__ == "__main__":
    from setup_info import *

    dataset = "argoverse_coco_finetune"
    test_data_loader = CustomCocoDetection(
        dataset_info[dataset]["test_root"], dataset_info[dataset]["test_json"], dataset,
    )
    print(test_data_loader.get_sequences_list())
    frames_info = test_data_loader.get_sequence(11)
    image, bboxes, classes = test_data_loader.get_frame(frames_info[0]["id"])
    # print(test_data_loader.coco_json["categories"])
    print(image.shape, bboxes.shape, classes.shape)
    # print(test_data_loader.get_frame(frames_info[0]["id"])    
    # print([x for x in test_data_loader.get_sequence(11)[:5]])
