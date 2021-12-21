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

def draw_patches(img,idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([1,2,0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)


def main():
    if  len(sys.argv) < 6:
        print ('Please pass image_folder augmentation_number output_image cpu/gpu batch_size')
        exit(0)
    data_path = sys.argv[1]
    augmentation_num = int(sys.argv[2])
    output_img = sys.argv[3]

    if(sys.argv[4] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False

    bs = int(sys.argv[5])
    nt = 1
    di = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

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
        images = fn.decoders.image(jpegs, file_root=data_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=True)
        # images = fn.decoders.image_random_crop(jpegs, device=decoder_device, output_type=types.RGB,
        #                                             device_memory_padding=device_memory_padding,
        #                                             host_memory_padding=host_memory_padding,
        #                                             random_aspect_ratio=[0.8, 1.25],
        #                                             random_area=[0.1, 1.0],
        #                                             num_attempts=100,
        #                                             file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        images = fn.resize(images, device=rali_device, resize_x=crop, resize_y=crop)
        # flip_coin = fn.random.coin_flip(probability=0.5)
        # images = fn.crop_mirror_normalize(images, device="gpu",
        #                                 output_dtype=types.FLOAT,
        #                                 output_layout=types.NCHW,
        #                                 crop=(crop, crop),
        #                                 mirror=flip_coin,
        #                                 image_type=types.RGB,
        #                                 mean=[0,0,0],
        #                                 std=[1,1,1])

        if augmentation_num == 0 or augmentation_num == 1:
                output = fn.resize(images,resize_x=crop_size,resize_y=crop_size)
        elif augmentation_num == 2:
                output = fn.rotate(images)
        elif augmentation_num == 3:
                output = fn.brightness(images)
        elif augmentation_num == 4:
                output = fn.gamma_correction(images)
        elif augmentation_num == 5:
                output = fn.contrast(images)
        elif augmentation_num == 6:
                output = fn.flip(images)
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
                output = fn.snp_noise(images)
        elif augmentation_num == 14:
                output = fn.snow(images)
        elif augmentation_num == 15:
                output = fn.rain(images)
        elif augmentation_num == 16:
                output = fn.fog(images)
        elif augmentation_num == 17:
                output = fn.pixelate(images)
        elif augmentation_num == 18:
                output = fn.exposure(images)
        elif augmentation_num == 19:
                output = fn.hue(images)
        elif augmentation_num == 20:
                output = fn.saturation(images)
        elif augmentation_num == 21:
                output = fn.color_twist(images)
        elif augmentation_num == 22:
                output = fn.crop_mirror_normalize(images,device="cpu",
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NCHW,
                                        crop=(crop, crop),
                                        image_type=types.RGB,
                                        mean=[0,0,0],
                                        std=[1,1,1])
        elif augmentation_num == 23:
                output = fn.nop(images)
        elif augmentation_num == 24:
                output = fn.centre_crop(images)
        elif augmentation_num == 25:
                output = fn.color_temp(images)
        elif augmentation_num == 26:
                output = fn.copy(images)
        elif augmentation_num == 27:
                output1 = fn.rotate(images)
                output2 = fn.fish_eye(output1)
                output3 = fn.fog(output2)
                pipe.set_outputs(output1,output2,output3)
                output_set=1
        elif augmentation_num == 28:
                output1 = fn.resize(images,resize_x=crop_size,resize_y=crop_size)
                output2 = fn.brightness(output1)
                output3 = fn.jitter(output2)
                pipe.set_outputs(output1,output2,output3)
                output_set=1
        elif augmentation_num == 29:
                output1 = fn.vignette(images)
                output2 = fn.blur(output1)
                pipe.set_outputs(output1,output2)
                output_set=1

        if output_set==0:
                pipe.set_outputs(output)


    pipe.build()

    data_loader = RALIClassificationIterator(pipe)
    epochs = 1
    cnt=0
    import timeit
    start = timeit.default_timer()
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\nImages:\n",it[0])
            print("\nLABELS:\n", it[1])
            for img in it[0]:
                cnt=cnt+1
                draw_patches(img,cnt)
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()


    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)
    print('Number of times loop iterates is:',cnt)

if __name__ == '__main__':
    main()
