# For documentation, please refer to "doc/tasks.md"

dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"
methodName=frcnn_ucb_256

CUDA_VISIBLE_DEVICES=2 python sim_streamer.py \
	--data-root "$dataDir/argoverse/data" \
	--annot-path "$dataDir/argoverse/annotations/train.json" \
	--fps 30 \
	--eta 0 \
    --model-name "faster_rcnn" \
    --dataset "argoverse" \
    --device "cuda:0" \
	--no-mask \
	--dynamic-schedule \
	--dynamic-schedule-type "rl-tradeoff" \
	--out-dir "$dataDir/Exp/ArgoVerse1.1/output/str_${methodName}/train" \
	--rl-config "frcnn_ucb_256" \
	--rl-model-load-folder "./models" \
	--rl-model-prefix "frcnn_ucb_256_latest"
	--overwrite \
