from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import random
import itertools
import os
from amd.rali.plugin.pytorch import RALIClassificationIterator
import cv2


from amd.rali.pipeline import Pipeline
import amd.rali.fn as fn
import amd.rali.types as types
import sys
import numpy as np

#Testing the fn import
# print(dir(fn))
# print(fn.__dict__)




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
        if self.display:
                try:
                        self.path= "OUTPUT_IMAGES_PYTHON/NEW_API/"+self.loader._reader+"/"
                        os.makedirs(self.path, exist_ok=True)
                except OSError as error:
                        print(error)
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        print('h:',self.h)
        print('w:',self.w)
        print('n:',self.n)

        self.p = (1 if color_format is types.GRAY else 3)
        print('p:',self.p)
        #self.out = np.empty(
                #(self.bs*self.n, self.p, int(self.h), self.w), dtype="uint8")
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

        #draw_patches(img[1], 0, 0)
        #Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        self.loader.GetImageId(self.image_id)
        # Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        self.count_batch = self.loader.GetBoundingBoxCount(
            self.bboxes_label_count)
        print("Count Batch:", self.count_batch)
        # 1D labels & bboxes array
        self.encoded_bboxes = np.zeros((self.count_batch*4), dtype="float32")
        self.encoded_labels = np.zeros(self.count_batch, dtype="int32")
        self.loader.copyEncodedBoxesAndLables(
            self.encoded_bboxes, self.encoded_labels)
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.loader.GetImgSizes(self.img_size)
        print("Image sizes:", self.img_size)

        encoded_bboxes_tensor = torch.tensor(self.encoded_bboxes).view(self.bs, -1, 4).contiguous()
        encodded_labels_tensor=  torch.tensor(self.encoded_labels).long().view(self.bs, -1)
        image_id_tensor = torch.tensor(self.image_id)
        image_size_tensor = torch.tensor(self.img_size).view(-1, self.bs, 2)
        num_images,_,_,_=np.shape(self.out)

        #Return images,bboxes,labels,image_id,image_size to the calling function
        if self.tensor_dtype == types.FLOAT:
            return torch.from_numpy(self.out), encoded_bboxes_tensor, encodded_labels_tensor, image_id_tensor, image_size_tensor,num_images
        elif self.tensor_dtype == types.FLOAT16:
            return torch.from_numpy(self.out.astype(np.float16)), encoded_bboxes_tensor, encodded_labels_tensor, image_id_tensor, image_size_tensor,num_images


    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img,idx, bboxes,path):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    print('Shape is:',img.shape)
    image = image.transpose([0,1,2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )

    _,htot ,wtot = img.shape
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)

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

def main():
    if  len(sys.argv) < 6:
        print ('Please pass image_folder annotation path augmentation_number output_image cpu/gpu batch_size <display=True/False>')
        exit(0)
    data_path = sys.argv[1]
    ann_path=sys.argv[2]
    augmentation_num = int(sys.argv[3])
    output_img = sys.argv[4]

    if(sys.argv[5] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False

    bs = int(sys.argv[6])
    display = int(sys.argv[7])
    nt = 1
    di = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    default_boxes = coco_anchors().numpy().flatten().tolist()
    pipe = Pipeline(batch_size=bs, num_threads=nt,device_id=di, seed=random_seed, rali_cpu=_rali_cpu)
    output_set=0
    local_rank = 0
    world_size = 1


    rali_cpu= True
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    crop_size = 300



    with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        images = fn.decoders.image_random_crop(jpegs, device=decoder_device, output_type=types.RGB,
                                                    device_memory_padding=device_memory_padding,
                                                    host_memory_padding=host_memory_padding,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100,
                                                    file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        images = fn.resize(images, device=rali_device, resize_x=crop, resize_y=crop)
        flip_coin = fn.random.coin_flip(probability=0.5)
        if display:
                images = fn.crop_mirror_normalize(images, device="gpu",
                                                output_dtype=types.FLOAT,
                                                output_layout=types.NCHW,
                                                crop=(crop, crop),
                                                mirror=flip_coin,
                                                image_type=types.RGB,
                                                mean=[0,0,0],
                                                std=[1,1,1])
        else:
                images = fn.crop_mirror_normalize(images, device="gpu",
                                                output_dtype=types.FLOAT,
                                                output_layout=types.NCHW,
                                                crop=(crop, crop),
                                                mirror=flip_coin,
                                                image_type=types.RGB,
                                                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                std=[0.229 * 255,0.224 * 255,0.225 * 255])

        if augmentation_num == 0 or augmentation_num == 1:
                output = fn.resize(images,resize_x=crop_size,resize_y=crop_size)
        elif augmentation_num == 2:
                output = fn.rotate(images,angle=90)
        elif augmentation_num == 3:
                output = fn.brightness(images)
        elif augmentation_num == 4:
                    #self.output = fn.gamma(images)
                    print('not yet implemented')
        elif augmentation_num == 5:
                output = fn.contrast(images,contrast=10)
        elif augmentation_num == 6:
                output = fn.flip(images,flip=1)
        elif augmentation_num == 7:
                output = fn.blur(images)
        elif augmentation_num == 8:
                images_hue = fn.hue(images)
                images_rotate = fn.rotate(images)
                output = fn.blend(images_hue, images_rotate)
        elif augmentation_num == 9:
                output = fn.warp_affine(images)
        elif augmentation_num == 10:
                output = fn.fish_eye(images)
        elif augmentation_num == 11:
                output = fn.vignette(images)
        elif augmentation_num == 12:
                output = fn.jitter(images)
        elif augmentation_num == 13:
                #self.output = fn.snpnoise(images)
                print('not yet implemented\n')
        elif augmentation_num == 14:
                output = fn.snow(images)
        elif augmentation_num == 15:
                output = fn.rain(images)
        elif augmentation_num == 16:
                output = fn.rain(images)
        elif augmentation_num == 17:
                output = fn.fog(images)
        elif augmentation_num == 18:
                output = fn.fog(images)
        elif augmentation_num == 19:
                output = fn.pixelate(images)
        elif augmentation_num == 20:
                output = fn.exposure(images)
        elif augmentation_num == 21:
                output = fn.hue(images)
        elif augmentation_num == 22:
                output = fn.saturation(images)
        elif augmentation_num == 23:
                output = fn.saturation(images)
        elif augmentation_num == 24:
                output = fn.color_twist(images)
                print('not yet implemented\n')
        elif augmentation_num == 25:
                    #output = fn.cropMirrorNormalize(images)
                print('not yet implemented\n')
        elif augmentation_num == 26:
                output1 = fn.rotate(images,angle=45)
                output2 = fn.fish_eye(output1)
                output3 = fn.fog(output2)
                pipe.set_outputs(output1,output2,output3)
                output_set=1
        elif augmentation_num == 27:
                output1 = fn.resize(images,resize_x=crop_size,resize_y=crop_size)
                output2 = fn.brightness(output1,brightness=2)
                output3 = fn.jitter(output2,preserve=True)
                pipe.set_outputs(output1,output2,output3)
                output_set=1
        elif augmentation_num == 28:
                output1 = fn.vignette(images)
                output2 = fn.blur(output1)
                pipe.set_outputs(output1,output2)
                output_set=1

        if output_set==0:
                pipe.set_outputs(output)


    pipe.build()
    if display:
        try:
                path= "OUTPUT_IMAGES_PYTHON/NEW_API/"+pipe._reader+"/"
                os.makedirs(path, exist_ok=True)
        except OSError as error:
                print(error)
    data_loader = RALIClassificationIterator(pipe , display=True)
    epochs = 2
    cnt=0
    import timeit
    start = timeit.default_timer()
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            cnt=cnt+1
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\nImages:\n",it[0])
            print("\nLABELS:\n", it[1])
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()


    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)
    print('Number of times loop iterates is:',cnt)

if __name__ == '__main__':
    main()
