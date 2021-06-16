from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import random
import itertools

from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types


import sys
import numpy as np


class COCOPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, data_dir, ann_dir, default_boxes,  crop, rali_cpu=True):
        super(COCOPipeline, self).__init__(batch_size, num_threads,
                                           device_id, seed=seed, rali_cpu=rali_cpu)
        self.input = ops.COCOReader(
            file_root=data_dir, annotations_file=ann_dir,random_shuffle=True,seed=seed)
        rali_device = 'cpu' if rali_cpu else 'gpu'
        decoder_device = 'cpu' if rali_cpu else 'mixed'
        # device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        # host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
        # 											device_memory_padding=device_memory_padding,
        # 											host_memory_padding=host_memory_padding,
        # 											random_aspect_ratio=[0.8, 1.25],
        # 											random_area=[0.1, 1.0],
        # 											num_attempts=100)
        self.decode = ops.ImageDecoder(
            device=decoder_device, output_type=types.RGB)
        self.crop = ops.SSDRandomCrop(num_attempts=5)
        self.decode_slice = ops.ImageDecoderSlice(device=decoder_device, output_type = types.RGB)
        self.random_bbox_crop = ops.RandomBBoxCrop(device="cpu",
                                       aspect_ratio=[0.5, 2.0],
                                       thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                       scaling=[0.3, 1.0],
                                       ltrb=True,
                                       allow_no_crop=True,
                                       num_attempts=1)
        self.res = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
        self.twist = ops.ColorTwist(device=rali_device)
        self.bbflip = ops.BBFlip(device=rali_device, ltrb=True)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 *255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
                                            #mean=[0, 0, 0],
                                            #std=[1, 1, 1])
        self.boxEncoder = ops.BoxEncoder(device=rali_device,
                                         criteria=0.5,
                                         anchors=default_boxes)
        self.cast = ops.Cast(device=rali_device, dtype=types.FLOAT)
        # Random variables
        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])
        self.coin_flip = ops.CoinFlip(probability=0.5)
        print('rali "{0}" variant'.format(rali_device))

    def define_graph(self):
        coin = self.coin_flip()
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        
        self.jpegs, self.bb, self.labels = self.input(name="Reader")
        # images = self.decode(self.jpegs)
        # images = self.crop(images)
        crop_begin, crop_size, bboxes, labels= self.random_bbox_crop(self.bb, self.labels)
        bboxes = self.bbflip(bboxes, horizontal=self.coin_flip)
        images = self.decode_slice(self.jpegs, crop_begin, crop_size)
        images = self.res(images)
        images = self.twist(images, saturation=saturation,
                            contrast=contrast, brightness=brightness, hue=hue)
        output = self.cmnp(images, mirror=coin)
        encoded_bboxes, encoded_labels = self.boxEncoder(bboxes, labels) # Encodes the bbox and labels ,input:"xywh" format output:"ltrb" format
        encoded_labels = self.cast(encoded_labels)
        return [output, encoded_bboxes, encoded_labels] #Encoded Bbox and labels output in "ltrb" format
        # return [output,  self.bb, self.labels]
        

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

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):

        # self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        if self.tensor_dtype == types.FLOAT:
            self.out = np.zeros(
                (self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype="float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out = np.zeros(
                (self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype="float16")

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

        if(types.NCHW == self.tensor_format):
            self.loader.copyToTensorNCHW(
                self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            self.loader.copyToTensorNHWC(
                self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))

        self.img_names_length = np.empty(self.bs, dtype="int32")
        self.img_names_size = self.loader.GetImageNameLen(
            self.img_names_length)
        print("Image name length:", self.img_names_size)
# Images names of a batch
        self.Img_name = self.loader.GetImageName(self.img_names_size)
        print("Image names in a batch ", self.Img_name)
# Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        self.count_batch = self.loader.GetBoundingBoxCount(
            self.bboxes_label_count)
        print("Count Batch:", self.count_batch)
# 1D labels array in a batch
        self.labels = np.zeros(self.count_batch, dtype="int32")
        self.loader.GetBBLabels(self.labels)
        print(self.labels)
# 1D bboxes array in a batch
        self.bboxes = np.zeros((self.count_batch*4), dtype="float32")
        self.loader.GetBBCords(self.bboxes)
        print(self.bboxes)
# Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.loader.GetImgSizes(self.img_size)
        print("Image sizes:", self.img_size)
        #img = torch.from_numpy(self.out)
        count = 0
        sum_count = 0
        for i in range(self.bs):
            count = self.bboxes_label_count[i]
            print("labels:", self.labels[sum_count: sum_count+count])
            print("bboxes:", self.bboxes[sum_count*4: (sum_count+count)*4])
            print("Image w & h:", self.img_size[i*2:(i*2)+2])
            print("Image names:", self.Img_name[i*16:(i*16)+12])
            self.img_name = self.Img_name[i*16:(i*16)+12]
            self.img_name = self.img_name.decode('utf_8')
            self.img_name = np.char.lstrip(self.img_name, chars='0')
            print("Image names:", self.img_name)
            self.label_2d_numpy = (self.labels[sum_count: sum_count+count])
            if(self.loader._BoxEncoder != True):
                self.label_2d_numpy = np.reshape(
                    self.label_2d_numpy, (-1, 1)).tolist()
            self.bb_2d_numpy = (self.bboxes[sum_count*4: (sum_count+count)*4])
            self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()
            # Draw images: make sure to revert the mean and std to 0 and 1 for displaying original images without normalization
            #draw_patches(img[i],self.img_name, self.bb_2d_numpy)
            if(self.loader._BoxEncoder == True):
                
                # Converting from "xywh" to "ltrb" format ,
                # where the values of l, t, r, b always lie between 0 & 1
                # Box Encoder input & output:
                # input : N x 4 , "xywh" format
                # output : 8732 x 4 , "xywh" format and normalized
                htot, wtot = 1, 1
                bbox_sizes = []
                i=0
                for (l,t,r,b) in self.bb_2d_numpy:
                    

                    bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
                    bbox_sizes.append(bbox_size)
                    i=i+1

                encoded_bboxes, encodded_labels = self.loader.encode(bboxes_in=bbox_sizes, labels_in=self.label_2d_numpy)
                if(self.loader._castLabels == True):
                    encodded_labels = encodded_labels.type(torch.FloatTensor)
                self.lis.append(encoded_bboxes)
                self.lis_lab.append(encodded_labels)
            else:
                self.lis_lab.append(self.label_2d_numpy)
                self.lis.append(self.bb_2d_numpy)
            sum_count = sum_count + count

        if (self.loader._BoxEncoder != True):
            self.target = self.lis
            self.target1 = self.lis_lab

            max_cols = max([len(row) for batch in self.target for row in batch])
            max_rows = max([len(batch) for batch in self.target])
            self.bb_padded = [
                batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
            self.bb_padded = torch.FloatTensor(
                [row + [0] * (max_cols - len(row)) for batch in self.bb_padded for row in batch])
            self.bb_padded = self.bb_padded.view(-1, max_rows, max_cols)
            # print(self.bb_padded)

            max_cols1 = max([len(row) for batch in self.target1 for row in batch])
            max_rows1 = max([len(batch) for batch in self.target1])
            self.labels_padded = [
                batch + [[0] * (max_cols1)] * (max_rows1 - len(batch)) for batch in self.target1]
            self.labels_padded = torch.LongTensor(
                [row + [0] * (max_cols1 - len(row)) for batch in self.labels_padded for row in batch])
            self.labels_padded = self.labels_padded.view(-1, max_rows1, max_cols1)
            # print(self.labels_padded)
        else:
            self.bb_padded = torch.stack(self.lis)
            self.labels_padded = torch.stack(self.lis_lab)

        if self.tensor_dtype == types.FLOAT:
            return torch.from_numpy(self.out), self.bb_padded, self.labels_padded
        elif self.tensor_dtype == types.FLOAT16:
            return torch.from_numpy(self.out.astype(np.float16)), self.bb_padded, self.labels_padded

    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        self.loader.raliResetLoaders()
        return self

def draw_patches(img,idx, bboxes):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([1,2,0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )
 
    _,htot ,wtot = img.shape
    
    for (l, t ,r,b) in bboxes:
        
        loc_ = [l, t ,r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]*wtot ),int( loc_[1] *htot)),(int((loc_[2] *wtot) ) ,int((loc_[3] *htot) )) , color, thickness)  
        #cv2.imwrite(str(idx)+"_"+"train"+".png", 255*image)
        cv2.imwrite(str(idx)+"_"+"train"+".png", image)

def main():
    if len(sys.argv) < 5:
        print('Please pass the folder image_folder Annotation_file cpu/gpu batch_size')
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
    crop_size = 300
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    def coco_anchors(): # Should be Tensor of floats in ltrb format - input - Mx4 where M="No of anchor boxes"
        fig_size = 300
        feat_size = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]
        
        # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
        scales = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        default_boxes = []
        fk = fig_size/np.array(steps)
        # size of feature and number of feature
        for idx, sfeat in enumerate(feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    default_boxes.append((cx, cy, w, h))
        dboxes = torch.tensor(default_boxes, dtype=torch.float)
        dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        dboxes_ltrb = dboxes.clone()
        dboxes_ltrb[:, 0] = dboxes[:, 0] - 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 1] = dboxes[:, 1] - 0.5 * dboxes[:, 3]
        dboxes_ltrb[:, 2] = dboxes[:, 0] + 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 3] = dboxes[:, 1] + 0.5 * dboxes[:, 3]
        
        return dboxes_ltrb

    dboxes = coco_anchors()

    pipe = COCOPipeline(batch_size=bs, num_threads=nt, device_id=di,seed = random_seed,
                        data_dir=image_path, ann_dir=ann_path, crop=crop_size, rali_cpu=_rali_cpu, default_boxes=dboxes)
    pipe.build()
    imageIterator = RALICOCOIterator(
        pipe, multiplier=pipe._multiplier, offset=pipe._offset)
    for i, it in enumerate(imageIterator, 0):
        print("**************", i, "*******************")
        print("**************starts*******************")
        print("\nBBOXES:\n", it[1])
        print("\nLABELS:\n", it[2])
        print("**************ends*******************")
        print("**************", i, "*******************")


if __name__ == '__main__':
    main()
