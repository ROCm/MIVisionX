#!/usr/bin/python

from rali import *
from rali_image_iterator import *
import cv2
import time

# Defining pipeline

class DataLoader(RaliGraph):
    def __init__(self, input_path, batch_size, input_color_format, affinity):
        RaliGraph.__init__(self, batch_size, affinity)
        self.setSeed(0)
        self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, 0)
        self.input = self.cropResize(self.jpg_img, 112,112, False, 0.6, -1, -1)

        self.warped = self.warpAffine(self.input,True)

        self.contrast_img = self.contrast(self.input,False)
        self.rain_img = self.rain(self.contrast_img, True)

        self.bright_img = self.brightness(self.input,False)
        self.temp_img = self.colorTemp(self.bright_img, True)

        self.exposed_img = self.exposure(self.input, False)
        self.vignette_img = self.vignette(self.exposed_img, False)
        self.fog_img = self.fog(self.vignette_img, False)
        self.snow_img = self.snow(self.fog_img, True)

        self.pixelate_img = self.pixelate(self.input, False)
        self.snp_img = self.SnPNoise(self.pixelate_img, False, 0.2)
        self.gamma_img = self.gamma(self.snp_img, True)

        self.rotate_img = self.rotate(self.input, False)
        self.jitter_img = self.jitter(self.rotate_img, True)

    def get_input_name(self, idx):
        return self.jpg_img.name(idx)

#%%

def main():
    batchSize = 1
    if  len(sys.argv) < 2:
        print 'Please pass the folder containing images as a command line argument'
        exit(0)

    input_path = sys.argv[1]

    # Pipeline creation
    loader = DataLoader(input_path, batchSize, ColorFormat.IMAGE_RGB24, Affinity.PROCESS_GPU)

    imageIterator = ImageIterator(loader)

    print "Input shape", loader.input.shape()

    print ('Pipeline created ...')

    print 'Image iterator created ... number of images', imageIterator.imageCount()

    start = time.time()


    for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
        cv2.imshow('image', cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
        print loader.get_input_name(0)
        cv2.waitKey(100)

    end = time.time()
    print 'Elapsed time1 ',(end - start)

if __name__ == '__main__':
    main()


#%%
