from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import torch
import random
import itertools

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

def draw_patches(img,idx, bboxes,display):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    print('Shape is:',img.shape)
    image = image.transpose([0,1,2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR )
 
    _,htot ,wtot = img.shape
    image = cv2.UMat(image).get()

    if display==1:
        cv2.imshow('Image',image)
        cv2.waitKey(0)

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
        print ('Please pass image_folder annotation path  cpu/gpu batch_size display')
        exit(0)
    image_path = sys.argv[1]
    ann_path=sys.argv[2]
    
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False

    bs = int(sys.argv[4])
    nt = 1
    di = 0
    display=int(sys.argv[5])
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop_size=300
    
    
    default_boxes = coco_anchors().numpy().flatten().tolist()
    pipe = Pipeline(batch_size=bs, num_threads=nt,device_id=di, seed=random_seed, rali_cpu=_rali_cpu)
    output_set=0

    with pipe:
        jpegs, bboxes, labels = fn.readers.coco(
            file_root=image_path, annotations_file=ann_path, random_shuffle=False, seed=random_seed) 
        decoded_images = fn.decoders.image(jpegs, output_type=types.RGB)
        images=fn.resize(decoded_images,resize_x=crop_size,resize_y=crop_size)
        flip_coin = fn.random.coin_flip(probability=0.5)
        bboxes = fn.bb_flip(bboxes, ltrb=True, horizontal=flip_coin)
        bboxes, labels = fn.box_encoder(bboxes, labels,
                                    criteria=0.5,
                                    anchors=default_boxes)
        images=fn.snow(images,snow=0.1)
        output=fn.jitter(images)
        pipe.set_outputs(output,bboxes,labels)
        
        
    #Create Iterator for COCO Pipeline
    data_loader = RALICOCOIterator(
        pipe, multiplier=pipe._multiplier, offset=pipe._offset,display=display)
    epochs = 1
    cnt=0
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            num_images=it[5]
            for ind in range(0,num_images):
                cnt=cnt+1
                draw_patches(it[0][ind],cnt,0,display)
        data_loader.reset()

if __name__ == '__main__':
    main()