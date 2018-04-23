export CUDA_VISIBLE_DEVICES=6
python convert_livingimages_to_TFRecodes.py --data_dir=. --output_path=. --train_proto=trainingButterfly_label_map.pbtxt --train_ratio=0.7

