
import sys
import types
import collections
import cv2
import numpy as np
from random import shuffle
from enum import Enum
import torch
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.plugin.pytorch import RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types


class ExternalInputIterator(object):
    def __init__(self, path, sinset_file_name, batch_size, mode, device_id, num_gpus):
        self.images_dir = path
        self.batch_size = batch_size
        self.mode = mode
        with open(self.images_dir + sinset_file_name, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        # whole data set size
        self.data_set_len = len(self.files) 
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i % self.n].split(' ')
            if self.mode == 'file_name':
              batch.append(self.images_dir + jpeg_filename)  # we can use numpy
            else:
              batch.append(np.fromfile(self.images_dir + jpeg_filename, dtype = np.uint8))  # we can use numpy
            labels.append(torch.tensor([int(label)], dtype = torch.uint8)) # or PyTorch's native tensors
            self.i += 1
        return (batch, labels)

    def __len__(self):
        return self.data_set_len

    next = __next__

class ExternalSourcePipe(Pipeline):
	def __init__(self, batch_size, num_threads, device_id, data_dir, file_name, image_size, rali_cpu = True):
		super(ExternalSourcePipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		world_size = 1
		local_rank = 0
		self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
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
		self.res = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
		self.rain = ops.Rain(rain=0.5)
		self.cmnp = ops.CropMirrorNormalize(device="gpu",
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=(crop, crop),
											image_type=types.RGB,
											mean=[0.485 * 255,0.456 * 255,0.406 * 255],
											std=[0.229 * 255,0.224 * 255,0.225 * 255])
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		rng = self.coin()
		self.jpegs, self.labels = self.input(name="Reader")
		images = self.decode(self.jpegs)
		images = self.rain(images)
		images = self.res(images)
		output = self.cmnp(images, mirror=rng)
		return [output, self.labels]

def main():
    if  len(sys.argv) < 5:
      print ('Please pass image_folder sinset_name cpu/gpu batch_size device_id')
      exit(0)
    _image_path = sys.argv[1]
    sinset_file_name = sys.argv[2]
    if(sys.argv[3] == "cpu"):
      _rali_cpu = True
    else:
      _rali_cpu = False
    bs = int(sys.argv[4])
    device_id = int(sys.argv[5])
    nt = 8
    crop_size = 224
    #call external_source_iterator

    pipe = ExternalSourcePipe(batch_size=bs, num_threads=nt, device_id=device_id, data_dir=_image_path, crop=crop_size, rali_cpu=_rali_cpu)
    #pipe.build()
    #world_size=1
    #imageIterator = RALI_iterator(pipe)

    for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
      cv2.imshow('image_batch', cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
      cv2.waitKey(10)

if __name__ == '__main__':
    main() 
