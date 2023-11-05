dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"

methodName=fcos_imagenet_vid_no_rl_baseline_600


python -m sap_toolkit.server \
	--data-root "$dataDir/imagenet-det+vid-argoverse-format/val" \
	--annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_val.json" \
	--out-dir "$dataDir/Exp/ImageNet-VID/output/${methodName}/val" \
	--overwrite \
	--eval-config "./config2.json" \
	--perf-factor 1.0

	# --data-root "$dataDir/argoverse/data" \
	# --annot-path "$dataDir/argoverse/annotations/val.json" \
	# --out-dir "$dataDir/Exp/Argoverse-HD/output/${methodName}/val" \

	# --data-root "$dataDir/imagenet-det+vid-argoverse-format/train" \
	# --annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_train.json" \
	# --out-dir "$dataDir/Exp/ImageNet-VID/output/${methodName}/train" \
