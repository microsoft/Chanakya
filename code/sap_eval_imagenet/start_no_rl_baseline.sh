dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"


methodName=fcos_imagenet_vid_no_rl_baseline_600

CUDA_VISIBLE_DEVICES=0 python forecast_no_rl_perf.py \
    --model-name "fcos" \
    --dataset "imagenet_vid_argoverse_format" \
    --annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_val.json" \
	--out-dir "$dataDir/Exp/ImageNet-VID/output/${methodName}/val" \
    --eval-config "./config2.json" \
    --no-mask \
    --dynamic-schedule \
	--perf-factor 1.0


    # --dataset "imagenet_vid_argoverse_format" \
    # --annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_train.json" \

    # --dataset "argoverse" \
    # --annot-path "$dataDir/argoverse/annotations/val.json" \
