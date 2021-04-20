
from amd.rali.plugin.tf import RALIClassificationIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import tensorflow as tf

class HybridPipe(Pipeline):
	def __init__(self, batch_size, num_threads, device_id, data_dir, crop, rali_cpu = True):
		super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		self.input = ops.TFRecordReader(path=data_dir, index_path = ""  , features={
		'image/encoded':tf.FixedLenFeature((), tf.string, ""),
		'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
		'image/class/text':tf.FixedLenFeature([ ], tf.string, ''),
		'image/object/bbox/xmin':tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymin':tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/xmax':tf.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymax':tf.VarLenFeature(dtype=tf.float32)
                })
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
		host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
		self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
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
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		inputs = self.input(name ="Reader")
		images = inputs["image/encoded"]
		labels = inputs["image/class/label"]
		images = self.decode(images)
		rng = self.coin()
		output = self.cmnp(images, mirror=rng)
		return [output, labels]

def main():
	if  len(sys.argv) < 4:
		print ('Please pass the TensorFlowrecord  cpu/gpu batch_size')
		exit(0)
	image_path = sys.argv[1]
	if(sys.argv[2] == "cpu"):
		_rali_cpu = True
	else:
		_rali_cpu = False
	bs = int(sys.argv[3])
	nt = 1
	di = 0
	crop_size = 224
	pipe = HybridPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=image_path, crop=crop_size, rali_cpu=_rali_cpu)
	pipe.build()
	imageIterator = RALIClassificationIterator(pipe)
	for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
		if(i % bs == 0):
			print("Comes to images ---in a batch,IMAGE TENSOR:",image_batch)
			print("Comes to images ---in a batch,LABEL TENSOR:",image_tensor)


if __name__ == '__main__':
    main()
