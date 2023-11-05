dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"


methodName=frcnn_esp_256_tracking_fixed_adv_480_300


CUDA_VISIBLE_DEVICES=3 python forecast_ours_tracking_perf.py \
    --model-name "faster_rcnn" \
    --dataset "argoverse" \
    --annot-path "$dataDir/argoverse/annotations/val.json" \
    --eval-config "./config.json" \
    --no-mask \
    --rl-config "frcnn_esp_256_tracking_fixed_adv_480_300" \
    --rl-model-folder "/FatigueDataDrive/HAMS-Edge/sim/models/frcnn_esp_256_tracking_fixed_adv_480_300_argoverse" \
    --rl-model-epoch 5 \
	--perf-factor 1.0
