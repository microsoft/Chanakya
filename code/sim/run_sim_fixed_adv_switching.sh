# For documentation, please refer to "doc/tasks.md"

dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"
methodName=frcnn_esp_imagenet_vid_256_r2_switching

CUDA_VISIBLE_DEVICES=2 python sim_streamer_switching.py \
	--data-root "$dataDir/imagenet-det+vid-argoverse-format/train" \
    --annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_train.json" \
	--fps 30 \
	--eta 0 \
    --dataset "imagenet_vid_argoverse_format" \
    --device "cuda:0" \
	--no-mask \
	--dynamic-schedule \
	--dynamic-schedule-type "rl-tradeoff" \
	--out-dir "$dataDir/Exp/ImageNet-VID-Sim/output/str_${methodName}/train" \
	--rl-config "frcnn_esp_imagenet_vid_256_r2_switching"   \
	--fixed-advantage-reward \
	--fixed-policy-results-folder "/FatigueDataDrive/HAMS-Edge-Datasets/Exp/ImageNet-VID/output/frcnn_imagenet_vid_no_rl_baseline_480_100/train" \
	--fixed-advantage-scale-factor 5 \
	--overwrite \

# --rl-model-load-folder "models/frcnn_esp_256_fixed_adv_480_300_argoverse" \
# --rl-model-prefix "epoch_7" \
