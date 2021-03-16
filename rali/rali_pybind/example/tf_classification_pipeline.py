from amd.rali.plugin.tf import RALIIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import tensorflow as tf

class HybridPipe(Pipeline):
	def __init__(self, feature_key_map, tfrecordreader_type, batch_size, num_threads, device_id, data_dir, crop, oneHotLabels=0, rali_cpu = True):
		super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		self.input = ops.TFRecordReader(path=data_dir, index_path = "", reader_type=tfrecordreader_type, user_feature_key_map=feature_key_map, 
			features={
				'image/encoded':tf.FixedLenFeature((), tf.string, ""),
				'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
				'image/filename':tf.FixedLenFeature((), tf.string, "")
			}
		)
		self._oneHotLabels = oneHotLabels
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
		host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
		self.decode = ops.ImageDecoderRandomCrop(user_feature_key_map=feature_key_map,
												device=decoder_device, output_type=types.RGB,
												device_memory_padding=device_memory_padding,
												host_memory_padding=host_memory_padding,
												random_aspect_ratio=[0.8, 1.25],
												random_area=[0.1, 1.0],
												num_attempts=100)
		self.cmnp = ops.CropMirrorNormalize(device="cpu",
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=(crop, crop),
											image_type=types.RGB,
											mean=[0.485 * 255,0.456 * 255,0.406 * 255],
											std=[0.229 * 255,0.224 * 255,0.225 * 255])
		self.one_hot_labels = ops.OneHot(num_classes=1000)
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		inputs = self.input(name ="Reader")
		images = inputs["image/encoded"]
		labels = inputs["image/class/label"]
		if(self._oneHotLabels == 1):
			labels = self.one_hot_labels(labels)
		images = self.decode(images)
		rng = self.coin()
		output = self.cmnp(images, mirror=rng)
		return [output, labels]

def main():
	if  len(sys.argv) < 5:
		print ('Please pass the <TensorFlowrecord> <cpu/gpu> <batch_size> <oneHotLabels=0/1>')
		exit(0)
	imagePath = sys.argv[1]
	if(sys.argv[2] == "cpu"):
		raliCPU = True
	else:
		raliCPU = False
	bs = int(sys.argv[3])
	oneHotLabel = int(sys.argv[4])
	nt = 1
	di = 0
	cropSize = 224
	TFRecordReaderType = 0
	featureKeyMap = {
		'image/encoded':'image/encoded',
		'image/class/label':'image/class/label',
		'image/filename':'image/filename'
	}

	pipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=bs, num_threads=nt, device_id=di, data_dir=imagePath, crop=cropSize, oneHotLabels=oneHotLabel, rali_cpu=raliCPU)
	pipe.build()
	
	imageIterator = RALIIterator(pipe)
	for i, (images_array, labels_array) in enumerate(imageIterator, 0):
		print("\n\n",i)
		print("\nIMAGES ARRAY:\n",images_array)
		print("\nLABELS ARRAY:\n",labels_array)
			
		

if __name__ == '__main__':
    main() 