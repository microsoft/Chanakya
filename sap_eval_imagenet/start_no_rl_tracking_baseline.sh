dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"


methodName=frcnn_argoverse_no_rl_tracking_baseline_600_600_2_ac

CUDA_VISIBLE_DEVICES=2 python forecast_no_rl_tracking_perf.py \
    --model-name "faster_rcnn" \
    --dataset "argoverse" \
    --annot-path "$dataDir/argoverse/annotations/val.json" \
    --eval-config "./config.json" \
    --no-mask \
	--perf-factor 1.0

# --dynamic-schedule \
