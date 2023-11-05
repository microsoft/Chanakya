dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"


methodName=frcnn_imagenet_vid_adascale_modi_dyna_baseline

CUDA_VISIBLE_DEVICES=0 python forecast_adascale_perf.py \
    --model-name "faster_rcnn" \
    --dataset "imagenet_vid_argoverse_format" \
    --annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_val.json" \
	--out-dir "$dataDir/Exp/ImageNet-VID/output/${methodName}/val" \
    --eval-config "./config.json" \
    --no-mask \
    --dynamic-schedule \
    --dynamic-schedule-type "rl-tradeoff" \
	--perf-factor 1.0

