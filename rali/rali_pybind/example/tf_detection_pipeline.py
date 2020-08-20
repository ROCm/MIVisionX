from amd.rali.plugin.tf import RALIIterator 
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import tensorflow as tf

class HybridPipe(Pipeline):
	def __init__(self, feature_key_map, tfrecordreader_type, batch_size, num_threads, device_id, data_dir, crop, rali_cpu = True):
		super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		self.input = ops.TFRecordReader(path=data_dir, index_path = "", reader_type=tfrecordreader_type, user_feature_key_map=feature_key_map, 
			features={
				'image/encoded':tf.FixedLenFeature((), tf.string, ""),
				'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
				'image/class/text':tf.FixedLenFeature([ ], tf.string, ''),
				'image/object/bbox/xmin':tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/ymin':tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/xmax':tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/ymax':tf.VarLenFeature(dtype=tf.float32),
				'image/filename':tf.FixedLenFeature((), tf.string, "")
			}
		)
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
		host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
		# self.decode = ops.ImageDecoder(user_feature_key_map=feature_key_map,
		# 										device=decoder_device, output_type=types.RGB)
		self.decode = ops.ImageDecoderRandomCrop(user_feature_key_map=feature_key_map,
												device=decoder_device, output_type=types.RGB,
												device_memory_padding=device_memory_padding,
												host_memory_padding=host_memory_padding,
												random_aspect_ratio=[0.8, 1.25],
												random_area=[0.1, 1.0],
												num_attempts=100)
		self.res = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
		self.cmnp = ops.CropMirrorNormalize(device="cpu",
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=(crop, crop),
											image_type=types.RGB,
											mean=[0.485 * 255,0.456 * 255,0.406 * 255],
											std=[0.229 * 255,0.224 * 255,0.225 * 255])
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		inputs = self.input(name ="Reader")
		images = inputs["image/encoded"]
		labels = inputs["image/class/label"]
		images = self.decode(images)
		images = self.res(images)
		rng = self.coin()
		output = self.cmnp(images, mirror=rng)
		return [output, labels]

def main():
	if  len(sys.argv) < 4:
		print ('Please pass the TensorFlowrecord  cpu/gpu batch_size')
		exit(0)
	imagePath = sys.argv[1]
	if(sys.argv[2] == "cpu"):
		raliCPU = True
	else:
		raliCPU = False
	bs = int(sys.argv[3])
	nt = 1
	di = 0
	cropSize = 224
	TFRecordReaderType = 1
	featureKeyMap = {
		'image/encoded':'image/encoded',
		'image/class/label':'image/object/class/label',
		'image/class/text':'image/object/class/text',
		'image/object/bbox/xmin':'image/object/bbox/xmin',
		'image/object/bbox/ymin':'image/object/bbox/ymin',
		'image/object/bbox/xmax':'image/object/bbox/xmax',
		'image/object/bbox/ymax':'image/object/bbox/ymax',
		'image/filename':'image/filename'
	}

	pipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=bs, num_threads=nt, device_id=di, data_dir=imagePath, crop=cropSize, rali_cpu=raliCPU) 
	pipe.build()
	
	imageIterator =  RALIIterator(pipe)
	for i, (images_array, bb, labels_array) in enumerate(imageIterator, 0):
		print("\nIMAGES ARRAY TYPE:\n",type(images_array))
		print("\nIMAGES ARRAY:\n",images_array)
		
		print("\nBBOXS ARRAY TYPE:\n",type(bb))
		print("\nBBOXS ARRAY:\n",bb)

		print("\nLABELS ARRAY TYPE:\n",type(labels_array))
		print("\nLABELS ARRAY:\n",labels_array)
			
		

if __name__ == '__main__':
	main() 