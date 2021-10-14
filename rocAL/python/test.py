# Copyright (c) 2018 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#!/usr/bin/python

from rali import *
from rali_image_iterator import *
from rali_common import *
import cv2
import time
import sys

# Defining pipeline

"""
class DataLoader(RaliGraph):
    def __init__(self, input_path, batch_size, input_color_format, affinity):
        RaliGraph.__init__(self, batch_size, affinity)
        self.setSeed(0)
        self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, 0)
        self.input = self.cropResize(self.jpg_img, 112,112, False, 0.6, -1, -1)
        #self.input = self.resize(self.jpg_img, 112,112,False)

        # self.warped = self.warpAffine(self.input,True)

        self.contrast_img = self.colorTwist(self.input,True)
        self.rain_img = self.rain(self.contrast_img, True)

        self.bright_img = self.brightness(self.input,False)
        self.temp_img = self.colorTemp(self.bright_img, True)

        self.exposed_img = self.exposure(self.input, False)
        self.vignette_img = self.vignette(self.exposed_img, False)
        self.fog_img = self.fog(self.vignette_img, False)
        self.snow_img = self.snow(self.fog_img, True)

        self.pixelate_img = self.pixelate(self.input, False)
        self.snp_img = self.SnPNoise(self.pixelate_img, False)
        self.gamma_img = self.gamma(self.snp_img, True)

        self.rotate_img = self.rotate(self.input, False)
        self.jitter_img = self.jitter(self.rotate_img, True)

    def get_input_name(self, idx):
        return self.jpg_img.name(idx)

#%%
"""
class DataLoader(RaliGraph):
    def __init__(self, input_path, rali_batch_size, model_batch_size, input_color_format, affinity, image_validation, h_img, w_img, raliMode, loop_parameter,
                     tensor_layout = TensorLayout.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=TensorDataType.FLOAT32):
        RaliGraph.__init__(self, rali_batch_size, affinity)
        self.validation_dict = {}
        self.process_validation(image_validation)
        self.setSeed(0)
        self.aug_strength = 0
        #params for contrast
        self.min_param = RaliIntParameter(0)
        self.max_param = RaliIntParameter(255)
        #param for brightness
        self.alpha_param = RaliFloatParameter(0.0)
        #param for colorTemp        
        self.adjustment_param = RaliIntParameter(0)
        #param for exposure
        self.shift_param = RaliFloatParameter(0.0)
        #param for SnPNoise
        self.sdev_param = RaliFloatParameter(0.0)
        #param for gamma
        self.gamma_shift_param = RaliFloatParameter(0.0)
        #param for rotate
        self.degree_param = RaliFloatParameter(0.0)

        #rocAL list of augmentation
        self.rali_list = None

        if model_batch_size == 16:
            if raliMode == 1:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, True)

                self.warped = self.warpAffine(self.input,True)

                self.contrast_img = self.contrast(self.input,True, self.min_param, self.max_param)
                self.rain_img = self.rain(self.input, True)

                self.bright_img = self.brightness(self.input,True, self.alpha_param)
                self.temp_img = self.colorTemp(self.input, True, self.adjustment_param)

                self.exposed_img = self.exposure(self.input, True, self.shift_param)
                self.vignette_img = self.vignette(self.input, True)
                self.blur_img = self.blur(self.input, True)
                self.snow_img = self.snow(self.input, True)

                self.pixelate_img = self.pixelate(self.input, True)
                self.snp_img = self.SnPNoise(self.input, True, self.sdev_param)
                self.gamma_img = self.gamma(self.input, True, self.gamma_shift_param)

                self.rotate_img = self.rotate(self.input, True, self.degree_param)
                self.flip_img = self.flip(self.input, True, 1)
                #self.jitter_img = self.jitter(self.input, True)
                
                self.blend_img = self.blend(self.input, self.contrast_img, True)
            elif raliMode == 2:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, True)

                self.warped = self.warpAffine(self.input,True)

                self.contrast_img = self.contrast(self.input,True, self.min_param, self.max_param)
                self.rain_img = self.rain(self.contrast_img, True)

                self.bright_img = self.brightness(self.input,True, self.alpha_param)
                self.temp_img = self.colorTemp(self.bright_img, True, self.adjustment_param)

                self.exposed_img = self.exposure(self.input, True, self.shift_param)
                self.vignette_itensor_dtypemg = self.vignette(self.exposed_img, True)
                self.blur_img = self.blur(self.input, True)
                self.snow_img = self.snow(self.blur_img, True)

                self.pixelate_img = self.pixelate(self.input, True)
                self.snp_img = self.SnPNoise(self.pixelate_img, True, self.sdev_param)
                self.gamma_img = self.gamma(self.input, True, self.gamma_shift_param)

                self.rotate_img = self.rotate(self.input, True, self.degree_param)
                self.flip_img = self.flip(self.input, True, 1)
                #self.jitter_img = self.jitter(self.rotate_img, True)

                self.blend_img = self.blend(self.rotate_img, self.warped, True)
            elif raliMode == 3:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, True)
                self.warped = self.warpAffine(self.input,True)

                self.contrast_img = self.contrast(self.input,True, self.min_param, self.max_param)
                self.rain_img = self.rain(self.warped, True)

                self.bright_img = self.brightness(self.input,True, self.alpha_param)
                self.temp_img = self.colorTemp(self.input, True, self.adjustment_param)

                self.exposed_img = self.exposure(self.input, True, self.shift_param)
                self.vignette_img = self.vignette(self.input, True)
                self.blur_img = self.blur(self.input, True)
                self.snow_img = self.snow(self.vignette_img, True)

                self.pixelate_img = self.pixelate(self.input, True)
                self.gamma_img = self.gamma(self.input, True, self.gamma_shift_param)
                self.snp_img = self.SnPNoise(self.gamma_img, True, self.sdev_param)

                self.rotate_img = self.rotate(self.input, True, self.degree_param)
                self.flip_img = self.flip(self.input, True, 1)
                #self.jitter_img = self.jitter(self.pixelate_img, True)

                self.blend_img = self.blend(self.snow_img, self.bright_img, True)
            elif raliMode == 4:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, True)

                for i in range(15):
                    self.copy_img = self.copy(self.input, True)

            elif raliMode == 5:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, True)

                for i in range(15):
                    self.nop_img = self.nop(self.input, True)
        elif model_batch_size == 64:
            if raliMode == 1:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, False)

                self.rot150_img = self.rotate(self.input, False, 150)
                self.flip_img = self.flip(self.input, False)
                self.rot45_img = self.rotate(self.input, False, 45)

                self.setof16_mode1(self.input, h_img, w_img)
                self.setof16_mode1(self.rot45_img, h_img, w_img)
                self.setof16_mode1(self.flip_img, h_img, w_img)
                self.setof16_mode1(self.rot150_img , h_img, w_img)
                
            elif raliMode == 2:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, False)

                #self.warpAffine2_img = self.warpAffine(self.input, False, [[1.5,0],[0,1],[None,None]])
                self.warpAffine1_img = self.warpAffine(self.input, False, [[0.5,0],[0,2],[None,None]]) #squeeze
                self.fishEye_img = self.fishEye(self.input, False)
                self.lensCorrection_img = self.lensCorrection(self.input, False, 1.5, 2)

                self.setof16_mode1(self.input, h_img, w_img)
                self.setof16_mode1(self.warpAffine1_img, h_img, w_img)
                self.setof16_mode1(self.fishEye_img, h_img, w_img)
                self.setof16_mode1(self.lensCorrection_img, h_img, w_img)

            elif raliMode == 3:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, False)

                self.colorTemp1_img = self.colorTemp(self.input, False, 10)
                self.colorTemp2_img = self.colorTemp(self.input, False, 20)
                self.warpAffine2_img = self.warpAffine(self.input, False, [[2,0],[0,1],[None,None]]) #stretch

                self.setof16_mode1(self.input, h_img, w_img)
                self.setof16_mode1(self.colorTemp1_img, h_img, w_img)
                self.setof16_mode1(self.colorTemp2_img, h_img, w_img)
                self.setof16_mode1(self.warpAffine2_img , h_img, w_img)
            elif raliMode == 4:
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, True)

                for i in range(63):
                    self.copy_img = self.copy(self.input, True)

            elif raliMode == 5: 
                self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
                self.input = self.resize(self.jpg_img, h_img, w_img, True)

                for i in range(63):
                    self.nop_img = self.nop(self.input, True)
        #rocAL iterator
        if self.build() != 0:
            raise Exception('Failed to build the augmentation graph')
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.w = self.getOutputWidth()
        self.h = self.getOutputHeight()
        self.b = self.getBatchSize()
        self.n = self.raliGetAugmentationBranchCount()
        color_format = self.getOutputColorFormat()
        self.p = (1 if color_format is ColorFormat.IMAGE_U8 else 3)
        height = self.h*self.n
        self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
        self.out_tensor = np.zeros(( self.b*self.n, self.p, self.h/self.b, self.w,), dtype = "float32")

    def get_input_name(self):
        size = self.raliGetImageNameLen(0)
        ret = ctypes.create_string_buffer(size)
        self.raliGetImageName(ret, 0)
        return ret.value

    def process_validation(self, validation_list):
        for i in range(len(validation_list)):
            name, groundTruthIndex = validation_list[i].decode("utf-8").split(' ')
            self.validation_dict[name] = groundTruthIndex

    def get_ground_truth(self):
        return self.validation_dict[self.get_input_name()]

    def setof16_mode1(self, input_image, h_img, w_img):
        self.resized_image = self.resize(input_image, h_img, w_img, True)

        self.warped = self.warpAffine(input_image,True)

        self.contrast_img = self.contrast(input_image,True, self.min_param, self.max_param)
        self.rain_img = self.rain(input_image, True)

        self.bright_img = self.brightness(input_image,True, self.alpha_param)
        self.temp_img = self.colorTemp(input_image, True, self.adjustment_param)

        self.exposed_img = self.exposure(input_image, True, self.shift_param)
        self.vignette_img = self.vignette(input_image, True)
        self.blur_img = self.blur(input_image, True)
        self.snow_img = self.snow(input_image, True)

        self.pixelate_img = self.pixelate(input_image, True)
        self.snp_img = self.SnPNoise(input_image, True, self.sdev_param)
        self.gamma_img = self.gamma(input_image, True, self.gamma_shift_param)

        self.rotate_img = self.rotate(input_image, True, self.degree_param)
        self.flip_img = self.flip(input_image, True, 1)
        #self.jitter_img = self.jitter(input_image, True)
        
        self.blend_img = self.blend(input_image, self.contrast_img, True)

    def updateAugmentationParameter(self, augmentation):
        #values for contrast
        self.aug_strength = augmentation
        min = int(augmentation*100)
        max = 150 + int((1-augmentation)*100)
        self.min_param.update(min)
        self.max_param.update(max)

        #values for brightness
        alpha = augmentation*1.95
        self.alpha_param.update(alpha)

        #values for colorTemp
        adjustment = (augmentation*99) if ((int(augmentation*100)) % 2 == 0) else (-1*augmentation*99)
        adjustment = int(adjustment)
        self.adjustment_param.update(adjustment)

        #values for exposure
        shift = augmentation*0.95
        self.shift_param.update(shift)

        #values for SnPNoise
        sdev = augmentation*0.7
        self.sdev_param.update(sdev)

        #values for gamma
        gamma_shift = augmentation*5.0
        self.gamma_shift_param.update(gamma_shift)



    def renew_parameters(self):
        curr_degree = self.degree_param.get()
        #values for rotation change
        degree = self.aug_strength * 100
        self.degree_param.update(curr_degree+degree)

    def start_iterator(self):
        self.reset()
        
    def get_next_augmentation(self):
        if self.raliIsEmpty() == 1:
            #raise StopIteration
            return -1
        self.renew_parameters()
        if self.run() != 0:
            #raise StopIteration
            return -1
        self.copyToNPArray(self.out_image)
        if(TensorLayout.NCHW == self.tensor_format):
            self.copyToTensorNCHW(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, self.tensor_dtype)

        else:
            self.copyToTensorNHWC(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, self.tensor_dtype)

        return self.out_image , self.out_tensor

    def get_rali_list(self, raliMode, model_batch_size):
        if model_batch_size == 16:
            if raliMode == 1:
                self.rali_list = raliList_mode1_16
            elif raliMode == 2:
                self.rali_list = raliList_mode2_16
            elif raliMode == 3:
                self.rali_list = raliList_mode3_16
            elif raliMode == 4:
                self.rali_list = raliList_mode4_16
            elif raliMode == 5:
                self.rali_list = raliList_mode5_16
        elif model_batch_size == 64:
            if raliMode == 1:
                self.rali_list = raliList_mode1_64
            elif raliMode == 2:
                self.rali_list = raliList_mode2_64
            elif raliMode == 3:
                self.rali_list = raliList_mode3_64
            elif raliMode == 4:
                self.rali_list = raliList_mode4_64
            elif raliMode == 5:
                self.rali_list = raliList_mode5_64
                
        return self.rali_list

def main():
    batchSize = 2
    if  len(sys.argv) < 3:
        print ('Please pass the folder containing images as a command line argument plus path to the labels file')
        exit(0)

    input_path = sys.argv[1]
    label_file_path = sys.argv[2]
    # Pipeline creation
    #loader = DataLoader(input_path, batchSize, ColorFormat.IMAGE_RGB24, Affinity.PROCESS_GPU)
    rali_batch_size = 1
    modelBatchSizeInt = 64

    fp = open(label_file_path, 'r')
    imageValidation = fp.readlines()
    fp.close()

    rali_mode = 1
    loop = False
    Mx = [1,1,1]
    Ax = [0,0,0]
    loader = DataLoader(input_path, rali_batch_size, modelBatchSizeInt, ColorFormat.IMAGE_RGB24, Affinity.PROCESS_CPU, imageValidation, 224, 224, rali_mode, loop, 
                                        TensorLayout.NCHW, False, Mx, Ax, TensorDataType.FLOAT32)

    imageIterator = ImageIterator(loader, tensor_dtype=TensorDataType.FLOAT32)

    print ("Input shape", loader.input.shape())

    print ('Pipeline created ...')

    print ('Image iterator created ... number of images', imageIterator.imageCount())

    start = time.time()


    for i, (image_batch, image_tensor) in enumerate(imageIterator, 0):
        cv2.imshow('image', cv2.cvtColor(image_batch, cv2.COLOR_RGB2BGR))
        print (loader.get_input_name(0))
        cv2.waitKey(0)

    end = time.time()
    print ('Elapsed time1 ',(end - start))

if __name__ == '__main__':
    main()


#%%
