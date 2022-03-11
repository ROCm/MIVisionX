from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import random
import itertools

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import numpy as np

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

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False):

        # self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
        self.display = display
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        self.hip_array = None
        if self.device == "cpu":
            if self.tensor_dtype == types.FLOAT:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float32)
            elif self.tensor_dtype == types.FLOAT16:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float16)
        else:
            torch_gpu_device = torch.device('cuda', self.device_id)
            if self.tensor_dtype == types.FLOAT:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float32, device=torch_gpu_device)
            elif self.tensor_dtype == types.FLOAT16:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float16, device=torch_gpu_device)

        #Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        # Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")


    def next(self):
        return self.__next__()

    def __next__(self):
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

        self.loader.copyToTensor(
            self.out, self.multiplier, self.offset, self.reverse_channels, self.tensor_format, self.tensor_dtype)

#Image id of a batch of images
        self.loader.GetImageId(self.image_id)
# Count of labels/ bboxes in a batch
        self.count_batch = self.loader.GetBoundingBoxCount(self.bboxes_label_count)
        print("Count Batch:", self.count_batch)
# 1D labels & bboxes array
        self.encoded_bboxes = np.zeros((self.count_batch*4), dtype="float32")
        self.encoded_labels = np.zeros(self.count_batch, dtype="int32")
        self.loader.copyEncodedBoxesAndLables(self.encoded_bboxes, self.encoded_labels)
# Image sizes of a batch
        self.loader.GetImgSizes(self.img_size)
        print("Image sizes:", self.img_size)

        encoded_bboxes_tensor = torch.tensor(self.encoded_bboxes).view(self.bs, -1, 4).contiguous()
        encodded_labels_tensor = torch.tensor(self.encoded_labels).long().view(self.bs, -1)
        image_id_tensor = torch.tensor(self.image_id)
        image_size_tensor = torch.tensor(self.img_size).view(-1, self.bs, 2)
        for i in range(self.bs):
            index_list = []
            actual_bboxes = []
            actual_labels = []
            for idx, x in enumerate(encodded_labels_tensor[i]):
                if x != 0:
                    index_list.append(idx)
                    actual_bboxes.append(encoded_bboxes_tensor[i][idx].tolist())
                    actual_labels.append(encodded_labels_tensor[i][idx].tolist())

            if self.display:
                img = self.out
                draw_patches(img[i], self.image_id[i], actual_bboxes, self.device)

        return (self.out), encoded_bboxes_tensor, encodded_labels_tensor, image_id_tensor, image_size_tensor

    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img,idx, bboxes, path_to_save_imgs):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([0,1,2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )

    wtot,htot ,_ = img.shape
    for (xc, yc ,w,h) in bboxes:
        l = xc - 0.5*(w)
        t = yc - 0.5*(h)
        r = xc + 0.5*(w)
        b = yc + 0.5*(h)
        loc_ = [l, t ,r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]*wtot ),int( loc_[1] *htot)),(int((loc_[2] *wtot) ) ,int((loc_[3] *htot) )) , color, thickness)
        cv2.imwrite(path_to_save_imgs+str(idx)+"_"+"train"+".png", image)

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
    default_boxes = coco_anchors().numpy().flatten().tolist()
    pipe = Pipeline(batch_size=bs, num_threads=1,device_id=0, seed=random_seed, rali_cpu=_rali_cpu)

    with pipe:
        jpegs, bboxes, labels = fn.readers.coco(
            file_root=image_path, annotations_file=ann_path, random_shuffle=False, seed=random_seed)
        crop_begin, crop_size, bboxes, labels = fn.random_bbox_crop(bboxes, labels,
                                                                device="cpu",
                                                                aspect_ratio=[0.5, 2.0],
                                                                thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                                                scaling=[0.3, 1.0],
                                                                bbox_layout="xyXY",
                                                                allow_no_crop=True,
                                                                num_attempts=50)
        images_decoded = fn.decoders.image_slice(jpegs, crop_begin, crop_size, device="mixed", output_type=types.RGB, file_root=image_path, annotations_file=ann_path, random_shuffle=False, seed=random_seed)
        # images_decoded = fn.decoders.image(jpegs, output_type=types.RGB, file_root=image_path, annotations_file=ann_path, random_shuffle=False, seed=random_seed)
        res_images = fn.resize(images_decoded, resize_x=300, resize_y=300)
        saturation = fn.uniform(range=[0.5, 1.5])
        contrast = fn.uniform(range=[0.5, 1.5])
        brightness = fn.uniform(range=[0.875, 1.125])
        hue = fn.uniform(range=[-0.5, 0.5])
        ct_images = fn.color_twist(res_images, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        # flip_coin = fn.random.coin_flip(probability=0.5)
        # bboxes = fn.bb_flip(bboxes, ltrb=True, horizontal=flip_coin)
        cmn_images = fn.crop_mirror_normalize(ct_images,
                                        crop=(300, 300),
                                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                        mirror=0,
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NCHW,
                                        pad_output=False)
        _, _ = fn.box_encoder(bboxes, labels,
                                    criteria=0.5,
                                    anchors=default_boxes)
        pipe.set_outputs(cmn_images)
    pipe.build()

    if( _rali_cpu):
        data_loader = RALICOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset,display=display, device="cpu")
    else:
        data_loader = RALICOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset,display=display, device="cuda")
    epochs = 2
    import timeit
    start = timeit.default_timer()

    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\n IMAGES : \n", it[0])
            print("\nBBOXES:\n", it[1])
            print("\nLABELS:\n", it[2])
            print("\nIMAGE ID:\n", it[3])
            print("\nIMAGE SIZE:\n", it[4])
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()
    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)


if __name__ == '__main__':
    main()






