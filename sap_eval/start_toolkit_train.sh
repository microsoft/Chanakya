dataDir="/FatigueDataDrive/HAMS-Edge-Datasets"

methodName=frcnn_argoverse_no_rl_baseline_train_360_1000

python -m sap_toolkit.server \
	--data-root "$dataDir/argoverse/data" \
	--annot-path "$dataDir/argoverse/annotations/train.json" \
	--overwrite \
	--out-dir "$dataDir/Exp/Argoverse-HD/output/${methodName}/train" \
	--eval-config "./config.json" \
	--perf-factor 1.0
