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

import sys
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types
import amd.rocal.fn as fn
import os
from parse_config import parse_args


def draw_patches(img, idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    args = parse_args()
    if args.rocal_gpu:
        image = img.cpu().numpy()
    else:
        image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if args.classification:
            cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/CAFFE2_READER/CLASSIFICATION/"+str(idx)+"_"+"train"+".png", image)
    else:
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/CAFFE2_READER/DETECTION/"+str(idx)+"_"+"train"+".png", image)


def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    _rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    _rocal_bbox = False if args.classification else True
    num_threads = args.num_threads
    local_rank =  args.local_rank
    random_seed = args.seed
    display = True if args.display else False
    device = "gpu" if args.rocal_gpu else "cpu"
    num_classes = len(next(os.walk(image_path))[1])
    print("num_classes:: ", num_classes)
    try:
        if args.classification:
            path= "OUTPUT_IMAGES_PYTHON/NEW_API/CAFFE2_READER/CLASSIFICATION/"
        else:
            path= "OUTPUT_IMAGES_PYTHON/NEW_API/CAFFE2_READER/DETECTION/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                    seed=random_seed, rocal_cpu=_rocal_cpu)
    with pipe:
        if _rocal_bbox:
            jpegs, labels, bboxes = fn.readers.caffe2(
                path=image_path, bbox=_rocal_bbox, random_shuffle=True)
        else:
            jpegs, labels = fn.readers.caffe2(
                path=image_path, bbox=_rocal_bbox, random_shuffle=True)
        images = fn.decoders.image(jpegs, output_type=types.RGB, path=image_path, random_shuffle=True)
        images = fn.resize(images, resize_x=224, resize_y=224)
        pipe.set_outputs(images)
    pipe.build()
    data_loader = ROCALClassificationIterator(pipe , display=display, device=device)

    # Training loop
    cnt = 0
    for epoch in range(1):  # loop over the dataset multiple times
        print("epoch:: ", epoch)
        if not _rocal_bbox:
            for i, (image_batch, labels) in enumerate(data_loader, 0):  # Classification
                if args.print_tensor:
                    sys.stdout.write("\r Mini-batch " + str(i))
                    print("Images", image_batch)
                    print("Labels", labels)
                for element in list(range(batch_size)):
                    cnt = cnt + 1
                    draw_patches(image_batch[element],cnt)
            data_loader.reset()
        else:
            for i, (image_batch, bboxes, labels) in enumerate(data_loader, 0):  # Detection
                if i == 0:
                    if args.print_tensor:
                        sys.stdout.write("\r Mini-batch " + str(i))
                        print("Images", image_batch)
                        print("Bboxes", bboxes)
                        print("Labels", labels)
                for element in list(range(batch_size)):
                    cnt = cnt + 1
                    draw_patches(image_batch[element],cnt)
            data_loader.reset()

    print("###############################################    CAFFE2 READER (CLASSIFCATION/ DETECTION)    ###############################################")
    print("###############################################    SUCCESS                                    ###############################################")

if __name__ == '__main__':
    main()
