import numpy as np
import cv2
from enum import Enum
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.plugin.pytorch import RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
class HybridTrainPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, data_dir, augmentation, crop, rali_cpu = True):
                super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
                self.aug_num = augmentation
                world_size = 1
                local_rank = 0
                self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
                rali_device = 'cpu' if rali_cpu else 'gpu'
                decoder_device = 'cpu' if rali_cpu else 'mixed'
                device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
                host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
                self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                                                                        device_memory_padding=device_memory_padding,
                                                                                                        host_memory_padding=host_memory_padding,
                                                                                                        random_aspect_ratio=[0.8, 1.25],
                                                                                                        random_area=[0.1, 1.0],
                                                                                                        num_attempts=100)
                self.resize = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
                #self.resizeCrop = ops.CropResize(crop, crop)
                self.exposure = ops.Exposure(exposure = 0.2)
                self.rotate = ops.Rotate()
                self.brightness = ops.Brightness()                
                self.gamma = ops.GammaCorrection()
                self.contrast = ops.Contrast()
                self.flip = ops.Flip()
                self.blur = ops.Blur(blur=0.5)
                self.blend = ops.Blend(blend=0.2)
                self.warpAffine = ops.WarpAffine()
                self.fisheye = ops.FishEye()
                self.vignette = ops.Vignette(vignette = 0.2)
                self.jitter = ops.Jitter()
                self.snpnoise = ops.SnPNoise(snpNoise = 0.5)
                self.snow = ops.Snow(snow=0.5)
                self.rain = ops.Rain(rain=0.5)
                self.fog = ops.Fog()
                
                self.pixelate = ops.Pixelate()
                self.exposure = ops.Exposure()
                self.hue = ops.Hue()
                self.saturation = ops.Saturation()
                self.colortwist = ops.ColorTwist()

                self.cropMirrorNormalize = ops.CropMirrorNormalize(device="gpu",
                                                                                        output_dtype=types.FLOAT,
                                                                                        output_layout=types.NCHW,
                                                                                        crop=(crop, crop),
                                                                                        image_type=types.RGB,
                                                                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                                                        std=[0.229 * 255,0.224 * 255,0.225 * 255])
                self.coin = ops.CoinFlip(probability=0.5)
                print('rali "{0}" variant'.format(rali_device))

        def define_graph(self):
                print("In define_graph function, augmentation = " + str(self.aug_num) + "\n")
                rng = self.coin()
                self.jpegs, self.labels = self.input(name="Reader")
                images = self.decode(self.jpegs)
                if self.aug_num != 0:
                    images = self.resize(images)
                
                
                if self.aug_num == 0:
                    output = self.resize(images)
                elif self.aug_num == 1:        
                    output = self.resize(images)
                elif self.aug_num == 2:
                    output = self.rotate(images)
                elif self.aug_num == 3:
                    output = self.brightness(images)
                elif self.aug_num == 4:
                    output = self.gamma(images)
                elif self.aug_num == 5:
                    output = self.contrast(images)
                elif self.aug_num == 6:
                    output = self.flip(images)
                elif self.aug_num == 7:
                    output = self.blur(images)
                elif self.aug_num == 8:
                    images_hue = self.hue(images)
                    images_rotate = self.rotate(images)
                    output = self.blend(images_hue, images_rotate)
                elif self.aug_num == 9:
                    output = self.warpAffine(images)
                elif self.aug_num == 10:
                    output = self.fisheye(images)
                elif self.aug_num == 11:
                    output = self.vignette(images)
                elif self.aug_num == 12:
                    output = self.jitter(images)
                elif self.aug_num == 13:
                    output = self.snpnoise(images)
                elif self.aug_num == 14:
                    output = self.snow(images)
                elif self.aug_num == 15:
                    output = self.rain(images)
                elif self.aug_num == 16:
                    output = self.rain(images)
                elif self.aug_num == 17:
                    output = self.fog(images)
                elif self.aug_num == 18:
                    output = self.fog(images)
                elif self.aug_num == 19:
                    output = self.pixelate(images)
                elif self.aug_num == 20:
                    output = self.exposure(images)
                elif self.aug_num == 21:
                    output = self.hue(images)
                elif self.aug_num == 22:
                    output = self.saturation(images)
                elif self.aug_num == 23:
                    output = self.saturation(images)
                elif self.aug_num == 24:
                    output = self.colortwist(images)
                elif self.aug_num == 25:
                    output = self.cropMirrorNormalize(images)
                
                
                
                return [output, self.labels]

def main():
        if  len(sys.argv) < 5:
                print ('Please pass image_folder augmentation_number output_image cpu/gpu batch_size')
                exit(0)
        _image_path = sys.argv[1]
        augmentation_num = int(sys.argv[2])
        output_img = sys.argv[3]
        if(sys.argv[4] == "cpu"):
                _rali_cpu = True
        else:
                _rali_cpu = False
        bs = int(sys.argv[5])
        nt = 1
        di = 0
        crop_size = 224
        pipe = HybridTrainPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=_image_path, augmentation=augmentation_num, crop=crop_size, rali_cpu=_rali_cpu)
        pipe.build()
        world_size=1
        imageIterator = RALI_iterator(pipe)


        for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
                cv2.imwrite(output_img, cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main() 
