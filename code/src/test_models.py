import os
import subprocess
import tidecv

base_location = "/FatigueDataDrive/HAMS-Edge/scale/saved_models/base/"
dataset_name = ["coco", "imagenet_vid"][1]
model_infos = {
    "coco" : [
        [
            "yolov3/yolov3_d53_mstrain-608_273e_coco.py", 
            "yolov3/yolov3_d53_mstrain-608_273e_coco-139f5633.pth", 
            "yolov3/outputs"
        ],
        [
            "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py", 
            "faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth", 
            "faster_rcnn/outputs"
        ],
        [
            "fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco.py", 
            "fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco_20200603-ed16da04.pth", 
            "fcos/outputs"
        ],
        [
            "retinanet/retinanet_r50_fpn_1x_coco.py",
            "retinanet/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
            "retinanet/outputs"
        ],
    ],
    "imagenet_vid" : [
        [
            "yolov3/yolov3_d53_mstrain-608_273e_imagenet_vid.py",
            "yolov3/yolov3_d53_mstrain-608_273e_imagenet_vid-epoch_7.pth",
            "yolov3/outputs"
        ],
        [
            "fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_mstrain640-800_imagenet_vid.py",
            "fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_mstrain640-800_imagenet_vid-epoch_9.pth",
            "fcos/outputs"
        ],
        [
            "faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid.py",
            "faster_rcnn/faster_rcnn_r50_fpn_1x_mstrain_imagenet_vid-epoch_10.pth",
            "faster_rcnn/outputs"
        ],
        
    ]
}
if dataset_name == "coco":
    dataset_gt_path = "/FatigueDataDrive/HAMS-Edge-Datasets/coco/annotations/instances_val2017.json"
elif dataset_name == "imagenet_vid":
    dataset_gt_path = "/FatigueDataDrive/HAMS-Edge-Datasets/imagenet-det+vid-coco-format/annotations/instances_val.json"

num_gpus = 4
if num_gpus <= 1:
    mmdetection_test_path = "/FatigueDataDrive/HAMS-Edge/extern/mmdetection/tools/test.py"
else:
    mmdetection_test_path = "/FatigueDataDrive/HAMS-Edge/extern/mmdetection/tools/dist_test.sh"

for model_info in model_infos[dataset_name]:
    print()
    print()

    model_config_path = os.path.join(base_location, model_info[0])
    model_checkpoint_path = os.path.join(base_location, model_info[1])
    
    model_out_path = os.path.join(base_location, model_info[2])

    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)

    model_results_pkl_path = os.path.join(model_out_path, "result_{}.pkl".format(dataset_name))
    model_results_coco_prefix_path = os.path.join(model_out_path, "result_coco_format_{}_dataset".format(dataset_name))
    model_results_coco_path = model_results_coco_prefix_path + ".bbox.json"

    if num_gpus <= 1:
        call_string = 'python {} {} {} --eval bbox --out {} --tmpdir /FatigueDataDrive/tmp/result_tmp --eval-options "jsonfile_prefix={}"'.format(mmdetection_test_path, model_config_path, model_checkpoint_path, model_results_pkl_path, model_results_coco_prefix_path)
    else:
        call_string = 'MKL_SERVICE_FORCE_INTEL=1 bash {} {} {} {} --eval bbox --out {} --tmpdir /FatigueDataDrive/tmp/result_tmp --eval-options "jsonfile_prefix={}"'.format(mmdetection_test_path, model_config_path, model_checkpoint_path, num_gpus, model_results_pkl_path, model_results_coco_prefix_path)

    subprocess.call(call_string, shell=True)

    tide = tidecv.TIDE()
    tide.evaluate(tidecv.datasets.COCO(dataset_gt_path), tidecv.datasets.COCOResult(model_results_coco_path), mode=tidecv.TIDE.BOX) # Use TIDE.MASK for masks
    tide.summarize()
    tide.plot(model_out_path)

    print()
    print()