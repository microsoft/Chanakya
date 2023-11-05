dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"


methodName=frcnn_adascale_modi_dyna_baseline

CUDA_VISIBLE_DEVICES=2 python forecast_adascale_perf.py \
    --model-name "faster_rcnn" \
    --dataset "argoverse" \
    --annot-path "$dataDir/argoverse/annotations/val.json" \
    --eval-config "./config.json" \
    --no-mask \
    --dynamic-schedule \
    --dynamic-schedule-type "rl-tradeoff" \
	--perf-factor 1.0

