# For documentation, please refer to "doc/tasks.md"

dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"
methodName=frcnn_esp_256_tracking_fixed_adv_480_300

CUDA_VISIBLE_DEVICES=3 python sim_streamer_tracking.py \
	--data-root "$dataDir/argoverse/data" \
	--annot-path "$dataDir/argoverse/annotations/train.json" \
	--fps 30 \
	--eta 0 \
	--model-name "faster_rcnn" \
    --dataset "argoverse" \
    --device "cuda:0" \
	--no-mask \
	--out-dir "$dataDir/Exp/ArgoVerse1.1/output/str_${methodName}/train" \
	--rl-config "frcnn_esp_256_tracking_fixed_adv_480_300" \
	--fixed-advantage-reward \
	--fixed-policy-results-folder "/FatigueDataDrive/HAMS-Edge-Datasets/Exp/Argoverse-HD/output/frcnn_argoverse_no_rl_baseline_train_480_300/train" \
	--fixed-advantage-scale-factor 5 \
	--overwrite \

	# --dynamic-schedule \
	# --dynamic-schedule-type "rl-tradeoff" \

# --rl-model-load-folder "models/frcnn_esp_256_fixed_adv_480_300_argoverse" \
# --rl-model-prefix "epoch_7" \
