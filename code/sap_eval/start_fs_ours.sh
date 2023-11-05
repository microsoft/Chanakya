dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"


methodName=frcnn_esp_256_fixed_adv_480_300


CUDA_VISIBLE_DEVICES=0 python forecast_ours_perf_aditya.py \
    --model-name "faster_rcnn" \
    --dataset "argoverse" \
    --annot-path "$dataDir/argoverse/annotations/val.json" \
    --eval-config "./config.json" \
    --no-mask \
    --dynamic-schedule \
    --rl-config "frcnn_esp_256_fixed_adv_480_300" \
    --rl-model-folder "/FatigueDataDrive/HAMS-Edge/sim/models/frcnn_esp_256_fixed_adv_480_300_argoverse" \
    --rl-model-epoch 5 \
	--perf-factor 1.0
