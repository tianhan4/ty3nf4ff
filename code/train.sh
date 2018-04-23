export CUDA_VISIBLE_DEVICES=4
MODEL_DIR=../models/faster_rcnn_resnet101_coco_11_06_2017/0423
PIPELINE_CONFIG=../config/faster_rcnn_resnet101_butterfly.config
python ./models/research/object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=${PIPELINE_CONFIG} \
	--train_dir=${MODEL_DIR}/train 

