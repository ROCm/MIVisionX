from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocAL.plugin.pytorch import RALI_iterator

from amd.rocAL.pipeline import Pipeline
import amd.rocAL.fn as fn
import amd.rocAL.types as types
import sys
import numpy as np
import cv2
import os

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    data_path = sys.argv[1]
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads,device_id=device_id, seed=random_seed, rali_cpu=_rali_cpu)
    local_rank = 0
    world_size = 1
    rali_cpu= True
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path)
        decoded_images = fn.decoders.image_random_crop(jpegs, device=decoder_device, output_type=types.RGB,
                                                    device_memory_padding=device_memory_padding,
                                                    host_memory_padding=host_memory_padding,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100,
                                                    file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        resized_images = fn.resize(decoded_images, device=rali_device, resize_x=crop, resize_y=crop)
        brightend_images = fn.brightness(resized_images)
        fish_eyed_image = fn.fish_eye(brightend_images)
        pipe.set_outputs(resized_images ,brightend_images, fish_eyed_image)

    pipe.build()
    imageIterator = RALI_iterator(pipe)
    epochs = 2
    import timeit
    start = timeit.default_timer() #Timer starts

    for epoch in range(int(epochs)):
        try:
            path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/"+"epoch"+str(epoch)+"/"
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            print(error)
        print("EPOCH:::::",epoch)
        for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
                cv2.imwrite(path+"output_images_"+str(i)+".jpg", cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
        imageIterator.reset()

    stop = timeit.default_timer() #Timer Stops
    print('\n Time: ', stop - start)

if __name__ == '__main__':
    main()
