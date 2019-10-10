
from  rali_common import *
import numpy as np

class ImageIterator:
    def __init__(self, pipeline,tensor_layout = TensorLayout.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0]):
        self.loader = pipeline
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        if pipeline.build() != 0:
            raise Exception('Failed to build the augmentation graph')
        self.w = pipeline.getOutputWidth()
        self.h = pipeline.getOutputHeight()
        self.b = pipeline.getBatchSize()
        self.n = pipeline.getOutputImageCount()
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is ColorFormat.IMAGE_U8 else 3)
        height = self.h*self.n
        #print ('h = ', height, 'w = ', self.w, 'p = ', self.p)
        self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
        self.out_tensor = np.zeros(( self.b*self.n, self.p, self.h/self.b, self.w,), dtype = "float32")
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.getReaminingImageCount() <= 0:
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration

        self.loader.copyToNPArray(self.out_image)
        if(TensorLayout.NCHW == self.tensor_format):
            self.loader.copyToTensorNCHW(self.out_tensor, self.multiplier, self.offset, self.reverse_channels)
        else:
            self.loader.copyToTensorNHWC(self.out_tensor, self.multiplier, self.offset, self.reverse_channels)

        return self.out_image , self.out_tensor

    def reset(self):
        self.loader.reset()

    def __iter__(self):
        self.loader.reset()
        return self

    def imageCount(self):
        return self.loader.getReaminingImageCount()
