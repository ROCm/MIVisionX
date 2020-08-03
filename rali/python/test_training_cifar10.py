# Copyright (c) 2018 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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



#%%
import sys
##############
import numpy as np
#import cv2
from rali_torch import *
from rali import *

#from enum import Enum
#from amd.rali.plugin.pytorch import RALIClassificationIterator
#from amd.rali.pipeline import Pipeline
#import amd.rali.ops as ops
#import amd.rali.types as types
PATH = '/home/rajy/pytorch_test/cifar_net_rali.pth'


class DataLoader(RaliGraph):
#define graph is not used for now: enable for testing with augmentation
	def graph(self, input, batch_size):
		warped = self.warpAffine(input,False)
		pixelate_img = self.pixelate(warped, False)
		temp_img = self.colorTemp(pixelate_img, False)
		vignette_img = self.vignette(temp_img, False)
		out0 = self.jitter(vignette_img, True)
		out0.set_labels(0)
		contrast_img = self.contrast(input,False)
		blur_img = self.blur(contrast_img, False)
		gamma_img = self.gamma(blur_img, False, 1.5)
		rotate_img = self.rotate(gamma_img, False)
		out1 = self.SnPNoise(rotate_img, True, 0.05)
		out1.set_labels(1)
		return out0, out1

	def __init__(self, path, file_prefix, batch_size, input_color_format, width, height, affinity):
		RaliGraph.__init__(self, batch_size, affinity)
		self.setSeed(0)
		self.CreateCifar10LabelReader(path, file_prefix)
		input0 = self.BinaryFileInput(path, input_color_format, True, width, height, file_prefix)
		self.out = input0


import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

classes = ('plane', 'car', 'bird', 'cat',
		   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def train():
	batchSize = 4
	nt = 1
	di = 0
	width = 32
	height = 32
	if  len(sys.argv) < 2:
		print ('Please pass the folder containing images as a command line argument')
		exit(0)

	image_path = str(sys.argv[1]).encode('ascii')
	file_prefix = str(sys.argv[2]).encode('ascii')
#	pipe = HybridTrainPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=image_path, crop=crop_size, rali_cpu=True)
#	pipe.build()
#	imageIterator = RALIClassificationIterator(pipe)
	# Pipeline creation
	loader = DataLoader(image_path, file_prefix, batchSize, ColorFormat.IMAGE_RGB_PLANAR, width, height, Affinity.PROCESS_CPU)

	torchIterator = PyTorchIterator(loader)

	net = Net()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

	# Training loop
	for epoch in range(2):  # loop over the dataset multiple times
		print("epoch:: ",epoch)
		running_loss = 0.0

		for i, (inputs, labels) in enumerate(torchIterator, 0):
			# get the inputs; data is a list of [inputs, labels]
			#sys.stdout.write("\r Mini-batch " + str(i))

			optimizer.zero_grad()

			outputs = net(inputs)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			print_interval = 1000
			if i % print_interval == (print_interval-1):
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / print_interval))
				running_loss = 0.0

	print('Finished Training')
	torch.save(net.state_dict(), PATH)		#save trained model

import torchvision.transforms as transforms
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='/home/rajy/pytorch_test/data', train=False,
									   download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										 shuffle=False, num_workers=2)

def test():
	net = Net()
	net.load_state_dict(torch.load(PATH))
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
#	import pdb
#	pdb.set_trace()
	train()
	test()
