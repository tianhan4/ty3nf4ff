r"""# convert all living images into one TFRecord file

Example usage:
	python convert_livingimages_to_TFRecords
	  --data_dir=path
	  --output_path=/...
	  --training_proto=...xxx.proto
"""
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import re
import random
import tensorflow as tf
import google.protobuf.text_format as pbt
import numpy
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from lxml import etree

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw butterfly training dataset.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecode')
flags.DEFINE_string('train_proto','trainingButterfly_label_map.pb','output label mapping file.' )
flags.DEFINE_float('train_ratio',0.7,'training data ratio per class' )
FLAGS = flags.FLAGS

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction=0.1

_decode_jpeg_data = tf.placeholder(dtype=tf.string)
_decode_jpeg = tf.image.decode_jpeg(_decode_jpeg_data, channels=3)

def to_tf_example(image,
	data,
	label_map_dict,
	height,
	width,
	image_subdirectory='JPEGImages',
	xml_subdirectory='Annotations'):
	"""Convert XML derived dict to tf.Example proto.
	
	Notice that this function normalizes the bounding box coordinates
	provided by the raw data.

	Concurrently generate trainingProto data"""
	
	xmin, ymin, xmax, ymax = [],[],[],[]
	classes, classes_text, truncated, poses = [],[],[],[]
	difficulties = []
	key = hashlib.sha256(image).hexdigest()
	if 'object' in data:
		for obj in data['object']:
			xmin.append(float(obj['bndbox']['xmin']) / width)
			ymin.append(float(obj['bndbox']['ymin']) / height)
			xmax.append(float(obj['bndbox']['xmax']) / width)
			ymax.append(float(obj['bndbox']['ymax']) / height)
			difficulties.append(int(obj['difficult']))
			text = obj['name']
			classes_text.append(text.encode('utf8'))
			#if text not in label_map_dict:
			#	label_map_dict[text] = len(label_map_dict) + 1
			#	assert label_map_dict[text] == len(label_map_dict.keys())
			classes.append(label_map_dict[text])
			truncated.append(int(obj['truncated']))
			poses.append(obj['pose'].encode('utf8'))
			

	example = tf.train.Example(features = tf.train.Features(feature={
		'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
		'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
		'image/height': dataset_util.int64_feature(height), 
		'image/width': dataset_util.int64_feature(width), 
		'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')), 
		'image/encoded': dataset_util.bytes_feature(image), 
		'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmin), 
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymin), 
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmax), 
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymax), 
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),  
		'image/object/class/label' : dataset_util.int64_list_feature(classes),
		'image/object/truncated' : dataset_util.int64_list_feature(truncated),
		'image/object/view' : dataset_util.bytes_list_feature(poses),
		'image/object/difficult' : dataset_util.int64_list_feature(difficulties)
		}))
	return example, classes[0]

#Use for generating proto for the first time.

"""
def generateTrainingProto(proto_file, label_to_dict):
	int_label_map = string_int_label_map_pb2.StringIntLabelMap()
	output = open(proto_file, "w")
	for key in label_to_dict.keys():
		butterfly = int_label_map.item.add()
		butterfly.id = label_to_dict[key]
		butterfly.name = key
	output.write(pbt.MessageToString(int_label_map))
	output.close()
"""


def getLabelArray(annotations_data_path, label_map_dict):
	labelArray =[0] * len(label_map_dict)
	for oneFile in os.listdir(annotations_data_path):
		xml_str = tf.gfile.GFile(os.path.join(annotations_data_path, oneFile),'r').read()
		xml = etree.fromstring(xml_str)
		data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

		if 'object' in data:
			for obj in data['object']:
				labelArray[label_map_dict[obj["name"]] - 1] += 1
	return numpy.array(labelArray)

def main(_):
	data_dir = FLAGS.data_dir
	proto_file = FLAGS.train_proto
	image_subdirectory='JPEGImages'
	xml_subdirectory='Annotations'
	writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,"butterfly_train.record"))
	writer2 = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,"butterfly_val.record"))
	
	logging.info('Reading from butterfly dataset.')
	training_data_path = os.path.join(data_dir, image_subdirectory)
	annotations_data_path = os.path.join(data_dir, xml_subdirectory)
	label_to_dict = label_map_util.get_label_map_dict(proto_file)
	random.seed(123)
	
	labelArray = getLabelArray(annotations_data_path, label_to_dict)
	labelCount = numpy.zeros_like(labelArray)
	labelLimit = (labelArray * FLAGS.train_ratio).astype(int)
	labelLimit[labelLimit==0] = 1
	print("labelArray:", labelArray, sum(labelArray))
	print("labelLimit:", labelLimit, sum(labelLimit))
	dirlist = os.listdir(training_data_path)
	random.shuffle(dirlist)

	with tf.Session(config=config) as sess:
		for oneFile in dirlist:
			name = re.match("(.+)\.jpg",oneFile)
			if not name:
				print("Matching file failed.")
				exit()
			name = name[1]
			training_file = os.path.join(training_data_path, name+".jpg")
			annotation_file = os.path.join(annotations_data_path, name+".xml")
			xml_str = tf.gfile.GFile(annotation_file,'r').read()
			xml = etree.fromstring(xml_str)
			data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
			print(data['filename'])
			image = tf.gfile.GFile(training_file, 'rb').read()
			image_array = sess.run(_decode_jpeg, feed_dict={_decode_jpeg_data: image})
			tf_example, label = to_tf_example(image, data, label_to_dict,image_array.shape[0], image_array.shape[1])
			print(label)
			labelCount[label - 1] += 1
			if labelCount[label - 1] <= labelLimit[label - 1]:
				writer.write(tf_example.SerializeToString())
			else:
				writer2.write(tf_example.SerializeToString())

	writer.close()
	writer2.close()
	#generateTrainingProto(proto_file, label_to_dict)


if __name__ == '__main__':
	tf.app.run()
