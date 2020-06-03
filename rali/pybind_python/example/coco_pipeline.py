from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import ctypes
import logging

from amd.rali.pipeline import Pipeline 
import amd.rali.ops as ops
import amd.rali.types as types


import sys
import numpy as np
from enum import Enum
import cv2
import time



class COCOPipeline(Pipeline):
	def __init__(self, batch_size, num_threads, device_id, data_dir,ann_dir, crop, rali_cpu = True):
		super(COCOPipeline, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)

     

		self.input = ops.COCOReader(file_root = data_dir, annotations_file = ann_dir)
		
		
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
											mirror=1,
											mean=[0.485 * 255,0.456 * 255,0.406 * 255],
											std=[0.229 * 255,0.224 * 255,0.225 * 255])
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		rng = self.coin()
		self.jpegs,self.bb, self.labels = self.input(name="Reader")
		images = self.decode(self.jpegs)
		images = self.res(images)
		output = self.cmnp(images)
		return [output, self.bb, self.labels]


class RALICOCOIterator(object):
    """
    COCO RALI iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """
    def __init__(self, pipelines, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
        
        # self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        
        self.loader = pipelines
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = self.loader._batch_size
        self.bbss=self.loader._batch_size
        # print("bs",self.bs)
        print("bbss",self.bbss)
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim=self.loader.getRemainingImages()
        print("____________REMAINING IMAGES____________:",self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        if self.tensor_dtype == types.FLOAT:
            self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float16")


    def next(self):
        return self.__next__()

    def __next__(self):
        print("In the next routine of COCO Iterator")
        if(self.loader.isEmpty()):
            timing_info = self.loader.Timing_Info()
            print("Load     time ::",timing_info.load_time)
            print("Decode   time ::",timing_info.decode_time)
            print("Process  time ::",timing_info.process_time)
            print("Transfer time ::",timing_info.transfer_time)
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration
        sum = 0
        self.lis =[] #Empty list for bboxes
        self.lis_lab=[] # Empty list of labels        

        if(types.NCHW == self.tensor_format):
            self.loader.copyToTensorNCHW(self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            self.loader.copyToTensorNHWC(self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))

        for idx in range(self.bs):
            sum=self.loader.GetBoundingBoxCount(idx)
            self.labels = np.zeros(sum,dtype = "int32")
            self.bboxes = np.zeros(sum*4,dtype = "float32" )
            self.loader.GetBBLabels(self.labels,idx)
            self.loader.GetBBCords(self.bboxes,idx)
            
            self.bb_2d_numpy = np.reshape(self.bboxes, (-1, 4)).tolist()
            self.label_2d_numpy = np.reshape(self.labels, (-1, 1)).tolist()
            
            self.lis.append(self.bb_2d_numpy)
            self.lis_lab.append(self.label_2d_numpy)

        self.target = self.lis
        self.target1 = self.lis_lab
        max_cols = max([len(row) for batch in self.target for row in batch])
        print("max_cols",max_cols)
        max_rows = max([len(batch) for batch in self.target])
        print("max_rows",max_rows)
        self.bb_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
        self.bb_padded = torch.FloatTensor([row + [0] * (max_cols - len(row)) for batch in self.bb_padded for row in batch])
        self.bb_padded = self.bb_padded.view(-1, max_rows, max_cols)
        print(self.bb_padded)

        max_cols1 = max([len(row) for batch in self.target1 for row in batch])
        print("max_cols1",max_cols1)
        max_rows1 = max([len(batch) for batch in self.target1])
        print("max_rows1",max_rows1)
        self.labels_padded = [batch + [[0] * (max_cols1)] * (max_rows1 - len(batch)) for batch in self.target1]
        self.labels_padded = torch.LongTensor([row + [0] * (max_cols1 - len(row)) for batch in self.labels_padded for row in batch])
        self.labels_padded = self.labels_padded.view(-1, max_rows1, max_cols1)
        print(self.labels_padded)

        if self.tensor_dtype == types.FLOAT:
            return torch.from_numpy(self.out),self.bb_padded, self.labels_padded
        elif self.tensor_dtype == types.FLOAT16:
            return torch.from_numpy(self.out.astype(np.float16)),self.bb_padded, self.labels_padded

    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        self.loader.raliResetLoaders()
        return self

def main():
	if  len(sys.argv) < 5:
		print ('Please pass the folder image_folder Annotation_file cpu/gpu batch_size')
		exit(0)
    
	image_path = sys.argv[1]
	ann_path = sys.argv[2]
	if(sys.argv[3] == "cpu"):
		_rali_cpu = True
	else:
		_rali_cpu = False
	bs = int(sys.argv[4])
	nt = 1
	di = 0
	crop_size = 224
	pipe = COCOPipeline(batch_size=bs, num_threads=nt, device_id=di, data_dir=image_path, ann_dir= ann_path, crop=crop_size, rali_cpu=_rali_cpu)        
	pipe.build()
	imageIterator = RALICOCOIterator(pipe)
	for i,it in enumerate(imageIterator,0):
		print("**************",i,"*******************")
		print(it[1])
		

    
if __name__ == '__main__':
	main()
