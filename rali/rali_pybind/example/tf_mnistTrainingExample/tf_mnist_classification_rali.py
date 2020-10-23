
from __future__ import print_function
from amd.rali.plugin.tf import RALIIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

############################### HYPER PARAMETERS FOR TRAINING ###############################
learning_rate = 0.001
image_size = 28
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
############################### HYPER PARAMETERS FOR TRAINING ###############################


def get_label_one_hot(label_ndArray):
	one_hot_vector_list = []
	for label in label_ndArray:
		one_hot_vector = np.zeros(num_classes)
		np.put(one_hot_vector, label - 1, 1)
		one_hot_vector_list.append(one_hot_vector)
	return one_hot_vector_list

# Create model
weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.nn.relu(layer_1)
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

#helper function not used in training
def decode(tfrecord_serialized):
	tfrecord_features = tf.parse_single_example(tfrecord_serialized, features={
		'image/height': tf.FixedLenFeature([], tf.int64),
		'image/width': tf.FixedLenFeature([], tf.int64),
		'image/class/label': tf.FixedLenFeature([], tf.int64),
		'image/raw': tf.FixedLenFeature([], tf.string),
	}, name='features')
	image = tf.decode_raw(tfrecord_features['image/raw'], tf.float32)
	image.set_shape([784])
	label = tf.cast(tfrecord_features['image/class/label'], tf.int32)
#	image_batch, label_batch = tf.train.batch([image, label], batch_size=bs)
	return image, label

#RALI pipeline
class HybridPipe(Pipeline):
	def __init__(self, feature_key_map, tfrecordreader_type, batch_size, num_threads, device_id, data_dir, crop, rali_cpu = True):
		super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		self.input = ops.TFRecordReader(path=data_dir, index_path = "", reader_type=tfrecordreader_type, user_feature_key_map=feature_key_map,
										features={
											'image/encoded':tf.FixedLenFeature((), tf.string, ""),
											'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
											'image/filename':tf.FixedLenFeature((), tf.string, "")
										},
										)
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		self.decode = ops.ImageDecoderRaw(user_feature_key_map=feature_key_map, device=decoder_device, output_type=types.RGB)
		#self.res = ops.Resize(device=rali_device, resize_x=crop[0], resize_y=crop[1])
		self.cmnp = ops.CropMirrorNormalize(device="cpu",
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=(crop,crop),
											image_type=types.GRAY,
											mean=[0 ,0,0],
											std=[255,255,255], mirror=0)
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		inputs = self.input(name ="Reader")
		images = inputs["image/encoded"]
		labels = inputs["image/class/label"]
		images = self.decode(images)
		#rng = self.coin()
		output = self.cmnp(images)
		return [output, labels]

# compute accuracy
def compute_accuracy(predictions, labels):
	correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
	return accuracy

def train_mnist_rali(data_path, _rali_cpu, batch_size):

	# setup keep_prob
	input_X = tf.placeholder('float32',shape = (batch_size,784))
	labels = tf.placeholder('float32',shape = (batch_size,10))
	logits = neural_net(input_X)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name="loss" )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	train_prediction = tf.nn.softmax(logits)
	accuracy = compute_accuracy(train_prediction, labels)
	#correct_label = tf.argmax(labels, 1)
	num_epochs = 10
	crop_size = 28
	TFRecordReaderType = 0
	featureKeyMap = {
		'image/encoded':'image_raw',
		'image/class/label':'label',
		'image/filename':''
	}

	trainPipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=batch_size, num_threads=1, device_id=0, data_dir=data_path+"/train", crop=crop_size, rali_cpu=_rali_cpu)
	valPipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=batch_size, num_threads=1, device_id=0, data_dir=data_path+"/val", crop=crop_size, rali_cpu=_rali_cpu)
	trainPipe.build()
	valPipe.build()
	trainIterator = RALIIterator(trainPipe)
	valIterator = RALIIterator(valPipe)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(num_epochs):
			print('\n\n----------------------------Training Model for Epoch: ', epoch, "-----------------------------------------------")
			epoch_loss = 0
			train_accuracy = 0
			for i, (image_train, label_train) in enumerate(trainIterator, 0):
				image_train_res = image_train.reshape(batch_size, 784)
				train_label_one_hot_list = get_label_one_hot(label_train)
				_, c, tacc = sess.run([optimizer, cost, accuracy], feed_dict={input_X:image_train_res, labels: train_label_one_hot_list})
				epoch_loss += c
				train_accuracy += tacc

			print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss, 'accuracy:',(train_accuracy*100)/i, 'count :', i)
			#run evaluation for every epoch
			mean_acc = 0
			print("\n\n----------------------------Evaluating Model ---------------------------------------------------------------")
			for j, (val_image_ndArray, val_label_ndArray) in enumerate(valIterator, 0):
				#val_image_ndArray_transposed = np.transpose(val_image_ndArray, [0, 2, 3, 1])
				val_image_ndArray_res = val_image_ndArray.reshape(batch_size, 784)
				val_label_one_hot_list = get_label_one_hot(val_label_ndArray)
				val_accuracy = sess.run(accuracy,
					#[optimizer, accuracy, prediction, correct_label, correct_pred],
					feed_dict={input_X: val_image_ndArray_res, labels: val_label_one_hot_list})
				mean_acc += val_accuracy
				#mean_loss = mean_loss + val_loss
				#num_correct_predicate = 0
				#for predicate in correct_predicate:
				#	if predicate == True:
				#		num_correct_predicate += 1
				#print ("Step :: %s\tTarget :: %s\tPrediction :: %s\tCorrect Predictions :: %s/%s\tValidation Loss :: %.2f\tValidation Accuracy :: %.2f%%\t" % (j, val_target, val_prediction, num_correct_predicate, len(correct_predicate), val_loss, (val_accuracy * 100)))
			mean_acc = (mean_acc * 100) / j
			#mean_loss = (mean_loss * 100)/ j
			print("\nSUMMARY:\nMean Accuracy :: %.2f%% count: %d" % (mean_acc, j))

def main():
	if  len(sys.argv) < 4:
		print ('Please pass mnistTFRecord_dir  cpu/gpu batch_size')
		exit(0)
	image_path = sys.argv[1]
	if(sys.argv[2] == "cpu"):
		_rali_cpu = True
	else:
		_rali_cpu = False
	bs = int(sys.argv[3])
	train_mnist_rali(image_path, _rali_cpu, bs)

if __name__ == '__main__':
	main()
