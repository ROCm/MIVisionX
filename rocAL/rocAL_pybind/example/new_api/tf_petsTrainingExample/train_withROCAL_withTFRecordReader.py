# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
from amd.rocal.plugin.tf import ROCALIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()


import numpy as np
import tensorflow_hub as hub





############################### CHANGE THESE GLOBAL VARIABLES APPROPRIATELY ###############################

RECORDS_DIR = 'tf_pets_records/'
NUM_CLASSES = 37
LEARNING_RATE = 0.005
NUM_TRAIN_STEPS = 2775
TRAIN_BATCH_SIZE = 8
EVAL_EVERY = 100
DATASET_DOWNLOAD_AND_PREPROCESS = False
RUN_ON_HOST = True

############################### CHANGE THESE GLOBAL VARIABLES APPROPRIATELY ###############################





######################################## NO CHANGES IN CODE NEEDED ########################################

TRAIN_RECORDS_DIR = RECORDS_DIR + 'train/'
VAL_RECORDS_DIR = RECORDS_DIR + 'val/'


def download_images():
	global TRAIN_RECORDS_DIR
	global VAL_RECORDS_DIR
	os.system("./download_and_preprocess_dataset.sh")
	TRAIN_RECORDS_DIR = "./tf_pets_records/train/"
	VAL_RECORDS_DIR = "./tf_pets_records/val/"

def create_model(features):
	global NUM_CLASSES
	layer = tf.compat.v1.layers.dense(inputs=features, units=NUM_CLASSES, activation=None)
	return layer

def get_label_one_hot(label_ndArray):
	global NUM_CLASSES
	one_hot_vector_list = []
	for label in label_ndArray:
		one_hot_vector = np.zeros(NUM_CLASSES)
		np.put(one_hot_vector, label - 1, 1)
		one_hot_vector_list.append(one_hot_vector)
	return one_hot_vector_list

def main():

	global NUM_CLASSES
	global LEARNING_RATE
	global NUM_TRAIN_STEPS
	global TRAIN_BATCH_SIZE
	global EVAL_EVERY
	global DATASET_DOWNLOAD_AND_PREPROCESS
	global TRAIN_RECORDS_DIR
	global VAL_RECORDS_DIR

	if DATASET_DOWNLOAD_AND_PREPROCESS == True:
		download_images()

	print("\n-----------------------------------------------------------------------------------------")
	print('TF records (train) are located in %s' % TRAIN_RECORDS_DIR)
	print('TF records (val) are located in %s' % VAL_RECORDS_DIR)
	print("-----------------------------------------------------------------------------------------\n")

	train_graph = tf.Graph()
	with train_graph.as_default():
		image_module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2')
		image_size = hub.get_expected_image_size(image_module)
		decoded_images = tf.compat.v1.placeholder(tf.float32, shape = [None, None, None, None])
		features = image_module(decoded_images)
		logits = create_model(features)
		labels = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
		cross_entropy_mean = tf.reduce_mean(input_tensor=cross_entropy)
		optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
		train_op = optimizer.minimize(loss=cross_entropy_mean)
		probabilities = tf.nn.softmax(logits)
		prediction = tf.argmax(input=probabilities, axis=1)
		correct_label = tf.argmax(input=labels, axis=1)
		correct_prediction = tf.equal(prediction, tf.argmax(input=labels, axis=1))
		accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

	crop_size = tuple(image_size)
	TFRecordReaderType = 0
	featureKeyMap = {
		'image/encoded':'image/encoded',
		'image/class/label':'image/object/class/label',
		'image/filename':'image/filename'
	}


	trainPipe = Pipeline(batch_size=TRAIN_BATCH_SIZE, num_threads=1, rocal_cpu=RUN_ON_HOST, tensor_layout = types.NHWC)
	with trainPipe:
		inputs = fn.readers.tfrecord(path=TRAIN_RECORDS_DIR, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
		features={
			'image/encoded':tf.io.FixedLenFeature((), tf.string, ""),
			'image/class/label':tf.io.FixedLenFeature([1], tf.int64,  -1),
			'image/filename':tf.io.FixedLenFeature((), tf.string, "")
			}
			)
		jpegs = inputs["image/encoded"]
		images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=TRAIN_RECORDS_DIR)
		resized = fn.resize(images, resize_x=crop_size[0], resize_y=crop_size[1])
		flip_coin = fn.random.coin_flip(probability=0.5)
		cmn_images = fn.crop_mirror_normalize(resized, crop=(crop_size[1], crop_size[0]),
                                              mean=[0,0,0],
                                              std=[255,255,255],
                                              mirror=flip_coin,
                                              output_dtype=types.FLOAT,
                                              output_layout=types.NHWC,
                                              pad_output=False)
		trainPipe.set_outputs(cmn_images)
	trainPipe.build()

	valPipe = Pipeline(batch_size=TRAIN_BATCH_SIZE, num_threads=1, rocal_cpu=RUN_ON_HOST, tensor_layout = types.NHWC)
	with valPipe:
		inputs = fn.readers.tfrecord(path=VAL_RECORDS_DIR, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
		features={
			'image/encoded':tf.io.FixedLenFeature((), tf.string, ""),
			'image/class/label':tf.io.FixedLenFeature([1], tf.int64,  -1),
			'image/filename':tf.io.FixedLenFeature((), tf.string, "")
			}
			)
		jpegs = inputs["image/encoded"]
		images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=VAL_RECORDS_DIR)
		resized = fn.resize(images, resize_x=crop_size[0], resize_y=crop_size[1])
		flip_coin = fn.random.coin_flip(probability=0.5)
		cmn_images = fn.crop_mirror_normalize(resized, crop=(crop_size[1], crop_size[0]),
                                              mean=[0,0,0],
                                              std=[255,255,255],
                                              mirror=flip_coin,
                                              output_dtype=types.FLOAT,
                                              output_layout=types.NHWC,
                                              pad_output=False)
		valPipe.set_outputs(cmn_images)
	valPipe.build()

	trainIterator = ROCALIterator(trainPipe)
	valIterator = ROCALIterator(valPipe)


	i = 0
	with tf.compat.v1.Session(graph = train_graph) as sess:
		sess.run(tf.compat.v1.global_variables_initializer())
		while i < NUM_TRAIN_STEPS:

			for t, (train_image_ndArray, train_label_ndArray) in enumerate(trainIterator, 0):
				train_label_one_hot_list = get_label_one_hot(train_label_ndArray)
				train_loss, _, train_accuracy = sess.run(
					[cross_entropy_mean, train_op, accuracy],
					feed_dict={decoded_images: train_image_ndArray, labels: train_label_one_hot_list})
				print ("Step :: %s\tTrain Loss :: %.2f\tTrain Accuracy :: %.2f%%\t" % (i, train_loss, (train_accuracy * 100)))
				is_final_step = (i == (NUM_TRAIN_STEPS - 1))
				if i % EVAL_EVERY == 0 or is_final_step:
					mean_acc = 0
					mean_loss = 0
					print("\n\n-------------------------------------------------------------------------------- BEGIN VALIDATION --------------------------------------------------------------------------------")
					for j, (val_image_ndArray, val_label_ndArray) in enumerate(valIterator, 0):
						val_label_one_hot_list = get_label_one_hot(val_label_ndArray)
						val_loss, val_accuracy, val_prediction, val_target, correct_predicate = sess.run(
							[cross_entropy_mean, accuracy, prediction, correct_label, correct_prediction],
							feed_dict={decoded_images: val_image_ndArray, labels: val_label_one_hot_list})
						mean_acc += val_accuracy
						mean_loss += val_loss
						num_correct_predicate = 0
						for predicate in correct_predicate:
							if predicate == True:
								num_correct_predicate += 1
						print ("Step :: %s\tTarget :: %s\tPrediction :: %s\tCorrect Predictions :: %s/%s\tValidation Loss :: %.2f\tValidation Accuracy :: %.2f%%\t" % (j, val_target, val_prediction, num_correct_predicate, len(correct_predicate), val_loss, (val_accuracy * 100)))
					mean_acc = (mean_acc * 100) / j
					print("\nSUMMARY:\nMean Loss :: %.2f\tMean Accuracy :: %.2f%%" % (mean_loss, mean_acc))
					print("\n-------------------------------------------------------------------------------- END VALIDATION --------------------------------------------------------------------------------\n\n")

				i = i + 1
				if i >= NUM_TRAIN_STEPS:
					break


if __name__ == '__main__':
	main()

######################################## NO CHANGES IN CODE NEEDED ########################################

