dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"

methodName=frcnn_esp_imagenet_vid_256_r2_switching

CUDA_VISIBLE_DEVICES=3 python forecast_ours_switching_perf.py \
    --dataset "argoverse" \
    --dataset "imagenet_vid_argoverse_format" \
	--annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_val.json" \
	--out-dir "$dataDir/Exp/ImageNet-VID/output/${methodName}/val" \
    --eval-config "./config.json" \
    --no-mask \
    --dynamic-schedule \
    --rl-config "frcnn_esp_imagenet_vid_256_r2_switching" \
    --rl-model-folder "/FatigueDataDrive/HAMS-Edge/sim/models/frcnn_esp_imagenet_vid_256_r2_switching_imagenet_vid_argoverse_format/" \
    --rl-model-prefix "epoch_1_n_seq_450" \
	--perf-factor 1.0
