################################################################################
#
# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import numpy as np
from rali_lib import *
from rali_image import *
from rali_parameter import *
from rali_common import *

RaliFlipAxis =  ('RALI_FLIP_HORIZONTAL','RALI_FLIP_VERTICAL')


class RaliGraph():
    def __init__(self, batch_size, affinity):
        self._lib = RaliLib()
        self.batch_size = batch_size
        self.output_images = []
        print('Going to call into createPipeline API')
        self.handle = self._lib.createPipeline(batch_size, affinity.value, 0, 1)
        if(self.handle != None):
            print('OK: Pipeline api found ')
        else:
            raise Exception('FAILED creating the pipeline')



    ImageSizeEvaluationPolicy = {
        'MAX_SIZE' : 0,
        'USER_GIVEN_SIZE' : 1,
        'MOST_FREQUENT_SIZE':2}

    """utility"""
    def validateFloatParameter(self, param):
        ret = param
        if param is not None:
            if isinstance(param, int) or isinstance(param, float):
                ret = RaliFloatParameter(param)
            else:
                if not isinstance(param, RaliFloatParameter) and not isinstance(param, RaliFloatUniformRand):
                    raise Exception('Unexpected parameter passed , should be a float type parameter')
            return ret.obj

        return ret

    def validateIntParameter(self, param):
        ret = param
        if param is not None:
            if isinstance(param, int):
                ret = RaliIntParameter(param)
            else:
                if not isinstance(param, RaliIntParameter) and not isinstance(param, RaliIntUniformRand):
                    raise Exception('Unexpected parameter passed , should be an int type parameter')
            return ret.obj

        return ret

    """ rali_api.h"""

    def run(self):
        return self._lib.run(self.handle)

    def __del__(self):
        self._lib.release(self.handle)

    def build(self):
        return self._lib.build(self.handle)

    """ rali_api_data_loader.h """

    def jpegFileInput(self, path, color_format,  is_output, loop = False, max_width = 0, max_height= 0, num_threads = 1):
        if max_width > 0 and max_height > 0:
            out = self._lib.raliJpegFileInput(self.handle, path, color_format.value, num_threads, is_output, loop, self.ImageSizeEvaluationPolicy['USER_GIVEN_SIZE'], max_width, max_height, 0)
        else:
            out = self._lib.raliJpegFileInput(self.handle, path, color_format.value, num_threads, is_output, loop, self.ImageSizeEvaluationPolicy['MOST_FREQUENT_SIZE'], 0, 0, 0)

        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def BinaryFileInput(self, path, color_format, is_output, width, height, file_prefix, loop = False):
        out = self._lib.raliBinaryFileInput(self.handle, path, color_format.value, is_output, width, height, file_prefix, loop)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img


    def reset(self):
        return self._lib.startOver(self.handle)


    """ rali_api_augmentation.h"""

    def resize(self, input, dest_width, dest_height, is_output):
        out = self._lib.raliResize(self.handle, input.obj, dest_width,dest_height, is_output)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def cropResize(self, input, dest_width, dest_height, is_output, area = None, x_center_drift = None, y_center_drift= None, aspect_ratio = None):
        param_area = self.validateFloatParameter( area)
        param_x_drift = self.validateFloatParameter( x_center_drift)
        param_y_drift = self.validateFloatParameter( y_center_drift)
        out = self._lib.raliCropResize(self.handle, input.obj,dest_width,dest_height, is_output, param_area,aspect_ratio, param_x_drift, param_y_drift)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def rotate(self, input, is_output,angle = None,  dest_width = 0, dest_height = 0):
        param_angle = self.validateFloatParameter(angle)
        if dest_width != 0 and dest_height != 0:
            out = self._lib.raliRotate(self.handle, input.obj, is_output, param_angle, dest_width, dest_height)
        else:
            out = self._lib.raliRotate(self.handle, input.obj, is_output, param_angle, 0, 0)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def brightness(self, input, is_output, alpha = None, beta = None):
        alpha_param = self.validateFloatParameter( alpha)
        beta_param = self.validateFloatParameter( beta)
        out = self._lib.raliBrightness(self.handle, input.obj, is_output, alpha_param, beta_param)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def gamma(self, input, is_output, alpha = None):
        param_alpha = self.validateFloatParameter(alpha)
        out = self._lib.raliGamma(self.handle, input.obj, is_output, param_alpha)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def contrast(self, input, is_output, min = None, max = None):
        min_param = self.validateIntParameter(min)
        max_param = self.validateIntParameter(max)
        out = self._lib.raliContrast(self.handle, input.obj, is_output, min_param, max_param)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def flip(self, input, is_output, flip_axis = 0):
        out = self._lib.raliFlip(self.handle, input.obj, flip_axis, is_output)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def blur(self, input, is_output, sdev = None):
        param_sdev = self.validateFloatParameter(sdev)
        out = self._lib.raliBlur(self.handle, input.obj, is_output, param_sdev)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def blend(self, input1, input2, is_output, ratio = None):

        param_ratio = self.validateFloatParameter(ratio)
        out = self._lib.raliBlend(self.handle, input1.obj,input2.obj, is_output, param_ratio)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def warpAffine(self, input, is_output, rotate_matrix = [[None,None],[None,None],[None,None]], dest_width = 0, dest_height = 0 ):
        x0 = self.validateFloatParameter(rotate_matrix[0][0])
        x1 = self.validateFloatParameter(rotate_matrix[0][1])
        y0 = self.validateFloatParameter(rotate_matrix[1][0])
        y1 = self.validateFloatParameter(rotate_matrix[1][1])
        o0 = self.validateFloatParameter(rotate_matrix[2][0])
        o1 = self.validateFloatParameter(rotate_matrix[2][1])
        if dest_width != 0 and dest_height != 0:
            out = self._lib.raliWarpAffine(self.handle, input.obj, is_output,dest_width, dest_height, x0, x1, y0, y1, o0, o1)
        else:
            out = self._lib.raliWarpAffine(self.handle, input.obj, is_output,0,0, x0, x1, y0, y1, o0, o1)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def fishEye(self, input, is_output):
        out = self._lib.raliFishEye(self.handle, input.obj, is_output)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def vignette(self, input, is_output, sdev = None):
        param_sdev = self.validateFloatParameter(sdev)
        out = self._lib.raliVignette(self.handle, input.obj, is_output, param_sdev)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def jitter(self, input, is_output, value = None):
        param_value = self.validateIntParameter(value)
        out = self._lib.raliJitter(self.handle, input.obj, is_output, param_value)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def SnPNoise(self, input, is_output, sdev = None):
        param_sdev = self.validateFloatParameter(sdev)
        out = self._lib.raliSnPNoise(self.handle, input.obj, is_output, param_sdev)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def snow(self, input, is_output, sdev = None):
        param_sdev = self.validateFloatParameter(sdev)
        out = self._lib.raliSnow(self.handle, input.obj, is_output, param_sdev)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def rain(self, input, is_output, rain_value = None):
        param_rain = self.validateFloatParameter(rain_value)
        out = self._lib.raliRain(self.handle, input.obj, is_output, param_rain, None, None, None )
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def colorTemp(self, input, is_output, adj_value = None):
        adj_value = self.validateIntParameter(adj_value)
        out = self._lib.raliColorTemp(self.handle, input.obj, is_output, adj_value)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def fog(self, input, is_output, fog_value = None):
        param_fog = self.validateFloatParameter(fog_value)
        out = self._lib.raliFog(self.handle, input.obj, is_output, param_fog)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def lensCorrection(self, input, is_output, strength = None, zoom = None):
        strength_param = self.validateFloatParameter(strength)
        zoom_param = self.validateFloatParameter(zoom)
        out = self._lib.raliLensCorrection(self.handle, input.obj, is_output, strength_param, zoom_param)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def pixelate(self, input, is_output):
        out = self._lib.raliPixelate(self.handle, input.obj, is_output)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def exposure(self, input, is_output, shift = None):
        param_shift = self.validateFloatParameter(shift)
        out = self._lib.raliExposure(self.handle, input.obj, is_output, param_shift)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def copy(self, input, is_output):
        out = self._lib.raliCopy(self.handle, input.obj, is_output)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    def nop(self, input, is_output):
        out = self._lib.raliNop(self.handle, input.obj, is_output)
        out_img = RaliImage(out)
        if is_output:
            self.output_images.append(out_img)
        return out_img

    """ rali_api_info.h """

    def getBatchSize(self):
        return self.batch_size

    def getReaminingImageCount(self):
        return self._lib.remainingImagesCount(self.handle)

    def raliIsEmpty(self):
        return self._lib.raliIsEmpty(self.handle)

    def raliGetAugmentationBranchCount(self):
        return self._lib.raliGetAugmentationBranchCount(self.handle)

    def raliGetImageName(self, buf, image_idx):
        return self._lib.raliGetImageName(self.handle, buf, image_idx)

    def raliGetImageNameLen(self, image_idx):
        return self._lib.raliGetImageNameLen(self.handle, image_idx)


    def setSeed(self, seed):
        self._lib.raliSetSeed(seed)

    def getSeed(self):
        return self._lib.raliGetSeed()

    def getOutputWidth(self):
        return self._lib.raliGetOutputWidth(self.handle)

    def getOutputHeight(self):
        return self._lib.raliGetOutputHeight(self.handle)

    def getOutputColorFormat(self):
        return self._lib.raliGetOutputColorFormat(self.handle)

    """ rali_api_meta_data.h"""
    def raliCreateTextFileBasedLabelReader(self, label_file):
        return self._lib.raliCreateTextFileBasedLabelReader(self.handle, label_file)

    def getImageLabels(self, buffer):
        return self._lib.raliGetImageLabels(self.handle, np.ascontiguousarray(buffer, dtype=np.int32))

    def CreateCifar10LabelReader(self, path, file_prefix):
        return self._lib.raliCreateCifar10LabelReader(self.handle, path, file_prefix)

    """ rali_api_data_transfer.h """

    def copyToNPArray(self, array ):
        out = np.frombuffer(array, dtype=array.dtype)
        self._lib.copyToOutput(self.handle, np.ascontiguousarray(out, dtype=array.dtype), array.size)

    def copyToTensorNHWC(self, array,  multiplier, offset, reverse_channels, tensor_dtype):
        out = np.frombuffer(array, dtype=array.dtype)
        if tensor_dtype == TensorDataType.FLOAT32:
            self._lib.copyToOutputTensor32(self.handle, np.ascontiguousarray(out, dtype=array.dtype), 0, multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), tensor_dtype)
        elif tensor_dtype == TensorDataType.FLOAT16:
            self._lib.copyToOutputTensor16(self.handle, np.ascontiguousarray(out, dtype=array.dtype), 0, multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), tensor_dtype)

    def copyToTensorNCHW(self, array,  multiplier, offset, reverse_channels, tensor_dtype):
        out = np.frombuffer(array, dtype=array.dtype)
        if tensor_dtype == TensorDataType.FLOAT32:
            self._lib.copyToOutputTensor32(self.handle, np.ascontiguousarray(out, dtype=array.dtype), 1, multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), tensor_dtype)
        elif tensor_dtype == TensorDataType.FLOAT16:
            self._lib.copyToOutputTensor16(self.handle, np.ascontiguousarray(out, dtype=array.dtype), 1, multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), tensor_dtype)

