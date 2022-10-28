# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
from parse_config import parse_args

class ROCALVideoIterator(object):
    """
    ROCALVideoIterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False ,sequence_length=3):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.batch_size = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
        self.display = display
        self.iter_num = 0
        self.sequence_length = sequence_length
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        self.out = np.empty(
                (self.batch_size*self.n,int(self.h/self.batch_size), self.w,self.p), dtype="ubyte")

    def next(self):
        return self.__next__()

    def __next__(self):
        self.iter_num +=1
        if(self.loader.isEmpty()):
            raise StopIteration
        if self.loader.run() != 0:
            raise StopIteration
        #Copy output from buffer to numpy array
        self.loader.copyImage(self.out)
        img = torch.from_numpy(self.out)
        #Display Frames in a video sequence
        if self.display:
            for batch_i in range(self.batch_size):
                draw_frames(img[batch_i], batch_i, self.iter_num)
        return img

    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

def draw_frames(img,batch_idx,iter_idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    # print('Shape is:',img.shape)
    image = image.transpose([0,1,2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )
    import os
    if not os.path.exists("OUTPUT_IMAGES_PYTHON/NEW_API/VIDEO_READER"):
        os.makedirs("OUTPUT_IMAGES_PYTHON/NEW_API/VIDEO_READER")
    image = cv2.UMat(image).get()
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/VIDEO_READER/"+"iter_"+str(iter_idx)+"_batch_"+str(batch_idx)+".png", image)

def main():
    #Args
    args = parse_args()
    video_path = args.video_path
    _rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    user_sequence_length = args.sequence_length
    display = args.display
    num_threads = args.num_threads
    random_seed = args.seed
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=args.local_rank, seed=random_seed, rocal_cpu=_rocal_cpu)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        images = fn.readers.video(device="gpu", file_root=video_path, sequence_length=user_sequence_length,
                              normalized=False, random_shuffle=False, image_type=types.RGB,
                              dtype=types.FLOAT, initial_fill=16, pad_last_batch=True, name="Reader")
        crop_size = (512,960)
        output_images = fn.crop_mirror_normalize(images,
                                            crop=crop_size,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            mirror=0,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            pad_output=False)
        pipe.set_outputs(output_images)
    # Build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALVideoIterator(
        pipe, multiplier=pipe._multiplier, offset=pipe._offset,display=display,sequence_length=user_sequence_length)
    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            if args.print_tensor:
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\n IMAGES : \n", it)
                print("**************ends*******************")
                print("**************", i, "*******************")
        data_loader.reset()
    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)


if __name__ == '__main__':
    main()






