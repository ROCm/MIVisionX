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

from rali_torch import *
from rali import *
import sys
class DataLoader(RaliGraph):

	def graph(self, input, batch_size):
		# warped = self.warpAffine(input,False)
		pixelate_img = self.pixelate(input, False)
		temp_img = self.colorTemp(pixelate_img, False)
		vignette_img = self.vignette(temp_img, False)
		out0 = self.jitter(vignette_img, True)
		out0.set_labels(0)
		contrast_img = self.contrast(input,False)
		blur_img = self.blur(contrast_img, False)
		gamma_img = self.gamma(blur_img, False)
		rotate_img = self.rotate(gamma_img, False)
		out1 = self.SnPNoise(rotate_img, True)
		out1.set_labels(1)
		return out0, out1

	def __init__(self, class_0_path,class_1_path, batch_size, input_color_format, affinity):
		RaliGraph.__init__(self, batch_size, affinity)
		self.setSeed(0)

		jpg_class_0 = self.jpegFileInput(class_0_path, input_color_format, False)
		input0 = self.cropResize(jpg_class_0, 224, 224, False, 0.6, -1, -1)
		#input0 = self.resize(jpg_class_0, 224,224,False)
		out0 = self.graph(input0, batch_size)


		jpg_class_1 = self.jpegFileInput(class_1_path, input_color_format, False)
		input1 = self.cropResize(jpg_class_1, 224, 224, False, 0.6, -1, -1)
		#input1 = self.resize(jpg_class_1, 224,224,False)
		out1 = self.graph(input1, batch_size)

		self.out = out0, out1

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ToyNet(nn.Module):
	def __init__(self):
		super(ToyNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv3 = nn.Conv2d(16, 64, 3)
		self.conv4 = nn.Conv2d(64, 256, 3)
		self.fc0 = nn.Linear(256 * 11*11, 2048)
		self.fc1 = nn.Linear(2048, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, 2) # Two classes only
	#self.m = nn.Softmax()


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
	print ('In the app')
	batchSize = 1
	if  len(sys.argv) < 3:
		print ('Please pass the folder containing images as a command line argument')
		exit(0)

	input_class_0 = sys.argv[1]
	input_class_1 = sys.argv[2]

	# Pipeline creation
	loader = DataLoader(input_class_0, input_class_1, batchSize, ColorFormat.IMAGE_RGB24, Affinity.PROCESS_GPU)

	torchIterator = PyTorchIterator(loader)

	print ('Torch iterator created ... number of images', torchIterator.imageCount() )

	# torchIterator = PyTorchIterator(loader)

	net = ToyNet()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

	# Training loop
	for epoch in range(100):  # loop over the dataset multiple times

		running_loss = 0.0

		for i, (image_batch , labels) in enumerate(torchIterator, 0):

			sys.stdout.write("\r Mini-batch " + str(i))

			optimizer.zero_grad()

			outputs = net(image_batch)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			print_interval = 100
			if i % print_interval == (print_interval-1):
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / print_interval))
				running_loss = 0.0

	print('Finished Training')

if __name__ == '__main__':
	main()
