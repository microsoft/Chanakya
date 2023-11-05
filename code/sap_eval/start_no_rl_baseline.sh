dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"


methodName=frcnn_imagenet_vid_no_rl_baseline_480_100

CUDA_VISIBLE_DEVICES=2 python forecast_no_rl_perf.py \
    --model-name "faster_rcnn" \
    --dataset "imagenet_vid_argoverse_format" \
    --annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_train.json" \
    --eval-config "./config.json" \
    --no-mask \
    --dynamic-schedule \
	--perf-factor 1.0

    # --dataset "imagenet_vid_argoverse_format" \
#     --annot-path "$dataDir/argoverse/annotations/val.json" \
