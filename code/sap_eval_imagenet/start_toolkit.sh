dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"

methodName=frcnn_esp_imagenet_vid_256_r2_switching


python -m sap_toolkit.server \
	--data-root "$dataDir/imagenet-det+vid-argoverse-format/val" \
	--annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_val.json" \
	--out-dir "$dataDir/Exp/ImageNet-VID/output/${methodName}/val" \
	--overwrite \
	--eval-config "./config.json" \
	--perf-factor 1.0

	# --data-root "$dataDir/argoverse/data" \
	# --annot-path "$dataDir/argoverse/annotations/val.json" \
	# --out-dir "$dataDir/Exp/Argoverse-HD/output/${methodName}/val" \

	# --data-root "$dataDir/imagenet-det+vid-argoverse-format/train" \
	# --annot-path "$dataDir/imagenet-det+vid-argoverse-format/annotations/instances_train.json" \
	# --out-dir "$dataDir/Exp/ImageNet-VID/output/${methodName}/train" \
