from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import random
import itertools

import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import numpy as np
from amd.rocal.pipeline import Pipeline

class RALICOCOIterator(object):
    """
    COCO RALI iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False):

        # self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
        self.display = display
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        self.out = np.empty(
                (self.bs*self.n,int(self.h/self.bs), self.w,self.p), dtype="ubyte")

    def next(self):
        return self.__next__()

    def __next__(self):
        print("In the next routine of COCO Iterator")
        if(self.loader.isEmpty()):
            timing_info = self.loader.Timing_Info()
            print("Load     time ::", timing_info.load_time)
            print("Decode   time ::", timing_info.decode_time)
            print("Process  time ::", timing_info.process_time)
            print("Transfer time ::", timing_info.transfer_time)
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration
        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels

         #Copy output from buffer to numpy array
        self.loader.copyImage(self.out)
#Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        self.loader.GetImageId(self.image_id)
# Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.loader.GetImgSizes(self.img_size)
        print("Image sizes:", self.img_size)

        image_id_tensor = torch.tensor(self.image_id)
        image_size_tensor = torch.tensor(self.img_size).view(-1, self.bs, 2)
        num_images,_,_,_=np.shape(self.out)

        if self.tensor_dtype == types.FLOAT:
            return torch.from_numpy(self.out),image_id_tensor, image_size_tensor, num_images
        elif self.tensor_dtype == types.FLOAT16:
            return torch.from_numpy(self.out.astype(np.float16)),image_id_tensor, image_size_tensor, num_images

    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img,idx,epoch,batch):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([0,1,2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )

    _,htot ,wtot = img.shape
    image = cv2.UMat(image).get()
    cv2.imwrite("epoch_"+str(epoch)+"_batch_"+str(batch)+"_idx_"+str(idx)+".png", image)

def main():
    if len(sys.argv) < 5:
        print('Please pass the folder image_folder Annotation_file cpu/gpu batch_size display(True/False)')
        exit(0)

    image_path = sys.argv[1]
    ann_path = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[4])
    display = sys.argv[5]
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    pipe = Pipeline(batch_size=bs, num_threads=num_threads, device_id=device_id, seed=random_seed, rali_cpu=_rali_cpu)

    with pipe:
        jpegs_dummy , labels, bboxes= fn.readers.coco(
            file_root=image_path, annotations_file=ann_path, random_shuffle=False, seed=random_seed)
        images_decoded = fn.decoders.image(jpegs_dummy, file_root=image_path, annotations_file=ann_path ,output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=True) # annotation path & source path has to be given as input to the decoder, shard_id & num_shards hads to be passed to the decoder
        res_images = fn.resize(images_decoded, resize_x=640, resize_y=640)
        fisheye_image = fn.fish_eye(res_images)
        brightness_image = fn.brightness(fisheye_image)
        pipe.set_outputs(images_decoded,res_images,fisheye_image,brightness_image)

    pipe.build()
    data_loader = RALICOCOIterator(
        pipe, multiplier=pipe._multiplier, offset=pipe._offset,display=display)
    epochs = 2
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\n IMAGES : \n", it[0])
            print("\nIMAGE ID:\n", it[1])
            print("\nIMAGE SIZE:\n", it[2])
            for ind in range(0,it[3]):
                print('\nimage number is:',ind+i);
                draw_patches(it[0][ind],it[3]*i+ind+1,epoch,i)
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()


if __name__ == '__main__':
    main()





