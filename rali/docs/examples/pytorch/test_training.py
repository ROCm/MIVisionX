################################################################################
#
# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import sys
##############
import numpy as np
import cv2
from enum import Enum
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import os


class HybridTrainPipe(Pipeline):
	def __init__(self, batch_size, num_threads, device_id, data_dir, crop, rali_cpu = True):
		super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
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
		images = self.res(images)
		output = self.cmnp(images, mirror=rng)
		return [output, self.labels]	

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ToyNet(nn.Module):
	def __init__(self,num_classes):
		super(ToyNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv3 = nn.Conv2d(16, 64, 3)
		self.conv4 = nn.Conv2d(64, 256, 3)
		self.fc0 = nn.Linear(256 * 11*11, 2048)
		self.fc1 = nn.Linear(2048, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, num_classes) # Two classes only
		self.m = nn.Softmax()


	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = self.pool(F.relu(self.conv4(x)))
		x = x.view(-1, 256 * 11 *11)
		x = F.relu(self.fc0(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x



def main():
	if  len(sys.argv) < 4:
		print ('Please pass image_folder cpu/gpu batch_size ')
		exit(0)
	if(sys.argv[2] == "cpu"):
		_rali_cpu = True
	else:
		_rali_cpu = False
	bs = int(sys.argv[3])
	nt = 1
	di = 0
	crop_size = 224
	image_path = sys.argv[1]
	pipe = HybridTrainPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=image_path, crop=crop_size, rali_cpu=_rali_cpu)        
	pipe.build()
	imageIterator = RALIClassificationIterator(pipe)
	num_classes = len(next(os.walk(image_path))[1])
	print("num_classes:: ",num_classes)
	net = ToyNet(num_classes)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

	# Training loop
	for epoch in range(10):  # loop over the dataset multiple times
		print("epoch:: ",epoch)
		running_loss = 0.0

		for i, (image_batch , labels) in enumerate(imageIterator, 0):

			sys.stdout.write("\r Mini-batch " + str(i))
			# print("Images",image_batch)
			# print("Labels",labels)

			optimizer.zero_grad()

			outputs = net(image_batch)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			print_interval = 10
			if i % print_interval == (print_interval-1):
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / print_interval))
				running_loss = 0.0

	print('Finished Training')

if __name__ == '__main__':
	main()
