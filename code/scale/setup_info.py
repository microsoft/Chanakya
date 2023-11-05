import os
import copy

base_dir = "/FatigueDataDrive/HAMS-Edge/scale"

models_info = {
    "coco": {
        "fcos": {
            "config_file": "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco.py",
            "checkpoint_file": "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco_20200603-ed16da04.pth",
            "regressors": {},
        },
        "faster_rcnn": {
            "config_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
            "checkpoint_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
            "regressors": {},
        },
        "cascade_rcnn": {"config_file": "", "checkpoint_file": "", "regressors": {},},
        "yolov3": {
            "config_file": "",
            "checkpoint_file": "",
            "regressors": {
                "scale": {
                    "train_scales": [(608, 608), (416, 416), (320, 320)],
                    "input_channels": 256,
                    "intermed_channels": 256,
                    "num_levels": 1,
                    "num_outputs": 1,
                    "outputs_activation": "none",
                    "filters": [1, 3],
                    "checkpoint_file": "saved_models/scale/yolov3/***",
                },
            },
        },
        "swin_tiny_mask_rcnn" :{
            "config_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",
            "checkpoint_file": "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/models/mask_rcnn_swin_tiny_patch4_window7.pth",
        },
        "swin_tiny_cascade_rcnn" :{
            "config_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/models/cascade_mask_rcnn_swin_tiny_patch4_window7.pth",
            "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py",
        }
    },
    "imagenet_vid": {
        "faster_rcnn": {
            "config_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid.py",
            "checkpoint_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid-epoch_10.pth",
            "regressors": {
                "scale": {
                    "train_scales": [
                        (2000, 600),
                        (2000, 480),
                        (2000, 360),
                        (2000, 240),
                    ],
                    "input_channels": 256,
                    "intermed_channels": 256,
                    "num_levels": 5,
                    "num_outputs": 1,
                    "outputs_activation": "none",
                    "filters": [1, 3],
                    "checkpoint_file": "saved_models/scale/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid.pth",
                },
                "complexity" : {
                    "train_scales" : [(2000, 600), (2000, 480), (2000, 360), (2000, 240)],
                    "input_channels" : 256,
                    "intermed_channels" : 256,
                    "num_levels" : 5,
                    "num_outputs" : 3,
                    "outputs_activation" : 'none',
                    "filters" : [1, 3],
                    # "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3/faster_rcnn_r50_fpn_1x_mstrain_argoverse-epoch_9.pth"
                    "checkpoint_file" : "saved_models/complexity/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid-epoch_10_5_0_11.pth"
                },
            },
        },
        "yolov3": {
            "config_file": "saved_models/base/yolov3/yolov3_d53_mstrain-608_273e_imagenet_vid.py",
            "checkpoint_file": "saved_models/base/yolov3/yolov3_d53_mstrain-608_273e_imagenet_vid-epoch_7.pth",
        },
        "fcos": {
            "config_file": "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_mstrain640-800_imagenet_vid.py",
            "checkpoint_file": "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_mstrain640-800_imagenet_vid-epoch_9.pth",
        },
        "swin_tiny_mask_rcnn" :{
            "config_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",
            "checkpoint_file": "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/models/mask_rcnn_swin_tiny_patch4_window7.pth",
        },
        "swin_tiny_cascade_rcnn" :{
            "config_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/models/cascade_mask_rcnn_swin_tiny_patch4_window7.pth",
            "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py",
        }

    },
    "argoverse" : {
        "faster_rcnn": {
            "config_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse.py",
            "checkpoint_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse-epoch_9.pth",
            "regressors": {
                "scale": {
                    "train_scales": [ (2000, 720), (2000, 640), (2000, 600), (2000, 560), (2000, 480), (2000, 360) ],
                    "input_channels": 256,
                    "intermed_channels": 256,
                    "num_levels": 5,
                    "num_outputs": 1,
                    "outputs_activation": "none",
                    "filters": [1, 3],
                    "checkpoint_file": "saved_models/scale/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse.pth",
                },
                "complexity" : {
                    "train_scales" : [(2000, 600), (2000, 480), (2000, 360), (2000, 240)],
                    "input_channels" : 256,
                    "intermed_channels" : 256,
                    "num_levels" : 5,
                    "num_outputs" : 3,
                    "outputs_activation" : 'none',
                    "filters" : [1, 3],
                    # "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3/faster_rcnn_r50_fpn_1x_mstrain_argoverse-epoch_9.pth"
                    "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3_train_full/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid_model_classification_1_3_49_18_18_0_2_2_12_2_8_1.pth"
                },
            }
        },
        "yolov3": {
            "config_file": "saved_models/base/yolov3/yolov3_d53_mstrain-608_273e_argoverse.py",
            "checkpoint_file": "saved_models/base/yolov3/yolov3_d53_mstrain-608_273e_argoverse-epoch_4.pth",
            "regressors": {
                "scale": {
                    "train_scales": [ (608, 608), (416, 416), (320, 320) ],
                    "input_channels": 256,
                    "intermed_channels": 256,
                    "num_levels": 5,
                    "num_outputs": 1,
                    "outputs_activation": "none",
                    "filters": [1, 3],
                    "checkpoint_file": "saved_models/scale/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse.pth",
                },
                "complexity" : {
                    "train_scales" : [(608, 608), (416, 416), (320, 320)],
                    "input_channels" : 256,
                    "intermed_channels" : 256,
                    "num_levels" : 5,
                    "num_outputs" : 3,
                    "outputs_activation" : 'none',
                    "filters" : [1, 3],
                    # "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3/faster_rcnn_r50_fpn_1x_mstrain_argoverse-epoch_9.pth"
                    "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3_train_full/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid_model_classification_1_3_49_18_18_0_2_2_12_2_8_1.pth"
                },
            }
        },
        "fcos": {
            "config_file": "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_mstrain640-800_argoverse.py",
            "checkpoint_file": "saved_models/base/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_mstrain640-800_argoverse-epoch_11.pth",
            "regressors": {
                "scale": {
                    "train_scales": [ (2000, 720), (2000, 640), (2000, 600), (2000, 560), (2000, 480), (2000, 360) ],
                    "input_channels": 256,
                    "intermed_channels": 256,
                    "num_levels": 5,
                    "num_outputs": 1,
                    "outputs_activation": "none",
                    "filters": [1, 3],
                    "checkpoint_file": "saved_models/scale/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse.pth",
                },
                "complexity" : {
                    "train_scales" : [(2000, 600), (2000, 480), (2000, 360), (2000, 240)],
                    "input_channels" : 256,
                    "intermed_channels" : 256,
                    "num_levels" : 5,
                    "num_outputs" : 3,
                    "outputs_activation" : 'none',
                    "filters" : [1, 3],
                    # "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3/faster_rcnn_r50_fpn_1x_mstrain_argoverse-epoch_9.pth"
                    "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3_train_full/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid_model_classification_1_3_49_18_18_0_2_2_12_2_8_1.pth"
                },
            }
        },
        "cascade_rcnn" : {
            "config_file": "saved_models/base/cascade_rcnn/cascade_rcnn_r101_fpn_1x_argoverse.py",
            "checkpoint_file": "saved_models/base/cascade_rcnn/epoch_10.pth",
            "regressors": {
                "scale": {
                    "train_scales": [ (2000, 720), (2000, 640), (2000, 600), (2000, 560), (2000, 480), (2000, 360) ],
                    "input_channels": 256,
                    "intermed_channels": 256,
                    "num_levels": 5,
                    "num_outputs": 1,
                    "outputs_activation": "none",
                    "filters": [1, 3],
                    "checkpoint_file": "saved_models/scale/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse.pth",
                },
                "complexity" : {
                    "train_scales" : [(2000, 600), (2000, 480), (2000, 360), (2000, 240)],
                    "input_channels" : 256,
                    "intermed_channels" : 256,
                    "num_levels" : 5,
                    "num_outputs" : 3,
                    "outputs_activation" : 'none',
                    "filters" : [1, 3],
                    # "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3/faster_rcnn_r50_fpn_1x_mstrain_argoverse-epoch_9.pth"
                    "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/complexity/complexity_regressor/saved_models/complexity/faster_rcnn_argoverse_coco_finetune_1_3_train_full/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid_model_classification_1_3_49_18_18_0_2_2_12_2_8_1.pth"
                },
            }

        },
        "swin_tiny_mask_rcnn" :{
            "config_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",
            "checkpoint_file": "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/models/mask_rcnn_swin_tiny_patch4_window7.pth",
        },
        "swin_tiny_cascade_rcnn" :{
            "config_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py",
            "checkpoint_file" : "/FatigueDataDrive/HAMS-Edge/extern/Swin-Transformer-Object-Detection/configs/swin/models/cascade_mask_rcnn_swin_tiny_patch4_window7.pth",
        }
    }
}

# format is different, dataset is same
models_info["imagenet_vid_argoverse_format"] = copy.deepcopy(
    models_info["imagenet_vid"]
)

models_info["argoverse_coco_finetune"] = copy.deepcopy(
    models_info["argoverse"]
)

# Let's use COCO pre-trained models for now
# models_info["argoverse"] = copy.deepcopy(models_info["coco"])
# models_info["argoverse"]["faster_rcnn"]["regressors"] = {
#     "scale": {
#         "train_scales": [(2000, 600), (2000, 480), (2000, 360), (2000, 240)],
#         "input_channels": 256,
#         "intermed_channels": 256,
#         "num_levels": 5,
#         "num_outputs": 1,
#         "outputs_activation": "none",
#         "filters": [1, 3],
#         "checkpoint_file": "saved_models/scale/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_pretrained_argoverse.pth",
#     },
# }

dataset_info = {
    "coco": {
        "train_root": "/FatigueDataDrive/HAMS-Edge-Datasets/coco/train2017",
        "test_root": "/FatigueDataDrive/HAMS-Edge-Datasets/coco/val2017",
        "train_json": "/FatigueDataDrive/HAMS-Edge-Datasets/coco/annotations/instances_train2017.json",
        "test_json": "/FatigueDataDrive/HAMS-Edge-Datasets/coco/annotations/instances_val2017.json",
    },
    ## This is DET + VID, used to train the object detector. The ordering of the VID frames doesn't matter here.
    ##
    ## Now, this folder contains
    ## 1. instances_train.json: this contains train sequences of VID sampled at every 15th frame along with DET images. No sequence data in this (ordering is irrelevant).
    ## 2. instances_val.json: this contains val sequences of VID sampled at every frame. No DET. No sequence data in this.
    ## 3. instances_val_15frames.json: this contains val sequences of VID sampled at every 15th frame. No DET. No sequence data in this.
    ##
    ## Why? Due to historical protocol reasons (DET : VID :: 2:1) mentioned by tracking the corresponding papers. First seen in Tubelet-CNN [TCSVT17].
    ## Cite: https://wang-zhe.me/welcome_files/papers/kangLYZ_tubelets.pdf
    "imagenet_vid": {
        "train_root": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-coco-format/train",
        "test_root": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-coco-format/val",
        "train_json": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-coco-format/annotations/instances_train.json",
        "test_json": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-coco-format/annotations/instances_val_15frames.json",
    },
    ## This is only VID. Used to evaluate at the level of sequence (so per seq evaluation). Ordering is given by "sid" and "fid".
    ## TODO's:
    ## 1. doesn't follow argoverse format to the dot as of yet. Need to add
    ##     (a) "sequences" whose value contains the array of "sid" and "seq" (sequence_name).
    ##     (b) "seq_dirs" which contains a dict with "sid" as key and "relative directory path" as value.
    ## 2. The "instances_train.json" file uses the wrong VID train split file, resulting in sampling every 15th frame.
    "imagenet_vid_argoverse_format": {
        "train_root": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-argoverse-format/train",
        "test_root": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-argoverse-format/val",
        "train_json": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-argoverse-format/annotations/instances_train.json",
        "test_json": "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-argoverse-format/annotations/instances_val.json",
    },
    "argoverse": {
        "train_root": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse/data/",
        "test_root": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse/data/",
        "train_json": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse/annotations/train.json",
        "test_json": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse/annotations/val.json",
    },
    ## What's going on. whyyyyyyy
    ##
    "argoverse_coco_finetune": {
        "train_root": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/data/train",
        "test_root": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/data/val",
        "train_json": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/annotations/train.json",
        "test_json": "/FatigueDataDrive/HAMS-Edge-Datasets/argoverse-coco-finetune/annotations/val.json",
    },
}

tracker_info = {
    "imagenet_vid": {
        "tracktor_faster_rcnn": {
            "config_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid.py",
            "checkpoint_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid-epoch_10.pth",
            "regressors": {
                "scale": {
                    "train_scales": [
                        (2000, 600),
                        (2000, 480),
                        (2000, 360),
                        (2000, 240),
                    ],
                    "input_channels": 256,
                    "intermed_channels": 256,
                    "num_levels": 5,
                    "num_outputs": 1,
                    "outputs_activation": "none",
                    "filters": [1, 3],
                    "checkpoint_file": "saved_models/scale/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid.pth",
                },
            },
        }
    },
    "argoverse": {
        "tracktor_faster_rcnn": {
            "config_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse.py",
            "checkpoint_file": "saved_models/base/faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_argoverse-epoch_9.pth",
            "regressors" : {}
        },
    }
}

tracker_info["imagenet_vid_argoverse_format"] = copy.deepcopy(
    tracker_info["imagenet_vid"]
)

for dataset in models_info:
    for model in models_info[dataset]:
        models_info[dataset][model]["config_file"] = os.path.join(
            base_dir, models_info[dataset][model]["config_file"]
        )
        models_info[dataset][model]["checkpoint_file"] = os.path.join(
            base_dir, models_info[dataset][model]["checkpoint_file"]
        )
        if "regressors" in models_info[dataset][model]:
            for regressor in models_info[dataset][model]["regressors"]:
                models_info[dataset][model]["regressors"][regressor][
                    "checkpoint_file"
                ] = os.path.join(
                    base_dir,
                    models_info[dataset][model]["regressors"][regressor][
                        "checkpoint_file"
                    ],
                )

for dataset in tracker_info:
    for model in tracker_info[dataset]:
        tracker_info[dataset][model]["config_file"] = os.path.join(
            base_dir, tracker_info[dataset][model]["config_file"]
        )
        tracker_info[dataset][model]["checkpoint_file"] = os.path.join(
            base_dir, tracker_info[dataset][model]["checkpoint_file"]
        )
        if "regressors" in tracker_info[dataset][model]:
            for regressor in tracker_info[dataset][model]["regressors"]:
                tracker_info[dataset][model]["regressors"][regressor][
                    "checkpoint_file"
                ] = os.path.join(
                    base_dir,
                    tracker_info[dataset][model]["regressors"][regressor][
                        "checkpoint_file"
                    ],
                )
