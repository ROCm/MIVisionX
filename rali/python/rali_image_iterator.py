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


from  rali_common import *
import numpy as np

class ImageIterator:
    def __init__(self, pipeline,tensor_layout = TensorLayout.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype = TensorDataType.FLOAT32):
        print("coming to Image itertor")
        self.loader = pipeline
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        if pipeline.build() != 0:
            raise Exception('Failed to build the augmentation graph')
        self.w = pipeline.getOutputWidth()
        self.h = pipeline.getOutputHeight()
        self.b = pipeline.getBatchSize()
        self.n = pipeline.raliGetAugmentationBranchCount()
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is ColorFormat.IMAGE_U8 else 3)
        height = self.h*self.n
        #print ('h = ', height, 'w = ', self.w, 'p = ', self.p)
        self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
        if self.tensor_dtype == TensorDataType.FLOAT32:        
            self.out_tensor = np.zeros(( self.b*self.n, self.p, self.h/self.b, self.w,), dtype = "float32")
        elif self.tensor_dtype == TensorDataType.FLOAT16:
            self.out_tensor = np.zeros(( self.b*self.n, self.p, self.h/self.b, self.w,), dtype = "float16")
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.getReaminingImageCount() <= 0:
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration

        self.loader.copyToNPArray(self.out_image)
        if(TensorLayout.NCHW == self.tensor_format):
            self.loader.copyToTensorNCHW(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            self.loader.copyToTensorNHWC(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        
        return self.out_image , self.out_tensor

    def reset(self):
        self.loader.reset()

    def __iter__(self):
        self.loader.reset()
        return self

    def imageCount(self):
        return self.loader.getReaminingImageCount()
