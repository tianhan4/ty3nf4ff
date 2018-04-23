# ty3nf4ff
config文件需要修改一下各个文件路径，train.sh, val.sh文件里面的路径和GPU设置酌情修改。
train.sh训练，val.sh evaluation，两个进程同时跑，后者周期性检查训练文件夹并在eval TFRecord上做evaluation.

# pre-trained文件
config:faster_rcnn_resnet101_butterfly.config
http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
