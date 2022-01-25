from amd.rali.plugin.pytorch import RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import datetime

class HybridTrainPipe(Pipeline):
	def __init__(self, batch_size, num_threads, device_id, data_dir, rali_cpu = True, prefetch_queue_depth = 2):
		super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, rali_cpu = rali_cpu, prefetch_queue_depth = prefetch_queue_depth)
		world_size = 1
		local_rank = 0
		resize_width = 300
		resize_height = 300
		self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
		self.res = ops.Resize(device=rali_device, resize_x=resize_width, resize_y=resize_height)
		self.rain = ops.Rain(rain=0.5)

	def define_graph(self):
		self.jpegs, self.labels = self.input(name="Reader")
		images = self.decode(self.jpegs)
		images = self.rain(images)
		output = self.res(images)
		return [output, self.labels]

def main():
	if  len(sys.argv) < 5:
		print ('Please pass image_folder cpu/gpu batch_size prefetch_queue_depth')
		exit(0)
	_image_path = sys.argv[1]
	if(sys.argv[2] == "cpu"):
		_rali_cpu = True
	else:
		_rali_cpu = False
	_batch_size = int(sys.argv[3])
	_prefetch_queue_depth = int(sys.argv[4])
	_num_threads = 5
	_device_id = 0
	pipe = HybridTrainPipe(batch_size = _batch_size, num_threads = _num_threads, device_id = _device_id,  data_dir=_image_path, rali_cpu=_rali_cpu, prefetch_queue_depth = _prefetch_queue_depth)
	pipe.build()
	imageIterator = RALI_iterator(pipe)
	start = datetime.datetime.now()
	for i in range(0, 10):
		for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
			continue
		imageIterator.reset()
	end = datetime.datetime.now()
	print("Time taken (averaged over 10 runs) ",int((end - start).total_seconds() * 1000),"milli seconds")

if __name__ == '__main__':
    main()
