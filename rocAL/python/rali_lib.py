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

import ctypes
from numpy.ctypeslib import ndpointer
from rali_common import *
import numpy as np


class RaliLib:
    def __init__(self):
        self.sharedlib = RALI_LIB_NAME
        self.lib = ctypes.cdll.LoadLibrary( self.sharedlib)

        """ rali_api.h """
        self.createPipeline = self.lib.raliCreate
        self.createPipeline.restype = ctypes.c_void_p
        self.createPipeline.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint]

        self.build = self.lib.raliVerify
        self.build.restype = ctypes.c_int
        self.build.argtypes = [ctypes.c_void_p]

        self.run = self.lib.raliRun
        self.run.restype = ctypes.c_int
        self.run.argtypes = [ctypes.c_void_p]

        self.release = self.lib.raliRelease
        self.release.restype = ctypes.c_int
        self.release.argtypes = [ctypes.c_void_p]

        """ rali_api_data_loaders.h"""

        self.raliJpegFileInput = self.lib.raliJpegFileSource
        self.raliJpegFileInput.restype = ctypes.c_void_p
        self.raliJpegFileInput.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]

        self.raliBinaryFileInput = self.lib.raliRawCIFAR10Source
        self.raliBinaryFileInput.restype = ctypes.c_void_p
        self.raliBinaryFileInput.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_bool, ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p, ctypes.c_bool]

        self.startOver = self.lib.raliResetLoaders
        self.startOver.restype = ctypes.c_int
        self.startOver.argtypes = [ctypes.c_void_p]

        """ rali_api_augmentation.h"""

        self.raliResize = self.lib.raliResize
        self.raliResize.restype = ctypes.c_void_p
        self.raliResize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool]

        self.raliCropResize = self.lib.raliCropResize
        self.raliCropResize.restype = ctypes.c_void_p
        self.raliCropResize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.raliCropResizeFixed = self.lib.raliCropResizeFixed
        self.raliCropResizeFixed.restype = ctypes.c_void_p
        self.raliCropResizeFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_bool, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]

        self.raliRotate = self.lib.raliRotate
        self.raliRotate.restype = ctypes.c_void_p
        self.raliRotate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_uint,  ctypes.c_uint]

        self.raliRotateFixed = self.lib.raliRotateFixed
        self.raliRotateFixed.restype = ctypes.c_void_p
        self.raliRotateFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool, ctypes.c_uint,  ctypes.c_uint]

        self.raliBrightness = self.lib.raliBrightness
        self.raliBrightness.restype = ctypes.c_void_p
        self.raliBrightness.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p]

        self.raliBrightnessFixed = self.lib.raliBrightnessFixed
        self.raliBrightnessFixed.restype = ctypes.c_void_p
        self.raliBrightnessFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_bool]

        self.raliGamma = self.lib.raliGamma
        self.raliGamma.restype = ctypes.c_void_p
        self.raliGamma.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliGammaFixed = self.lib.raliGammaFixed
        self.raliGammaFixed.restype = ctypes.c_void_p
        self.raliGammaFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliContrast = self.lib.raliContrast
        self.raliContrast.restype = ctypes.c_void_p
        self.raliContrast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p]

        self.raliContrastFixed = self.lib.raliContrastFixed
        self.raliContrastFixed.restype = ctypes.c_void_p
        self.raliContrastFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int]

        self.raliFlip = self.lib.raliFlipFixed
        self.raliFlip.restype = ctypes.c_void_p
        self.raliFlip.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]

        self.raliBlur = self.lib.raliBlur
        self.raliBlur.restype = ctypes.c_void_p
        self.raliBlur.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliBlurFixed = self.lib.raliBlurFixed
        self.raliBlurFixed.restype = ctypes.c_void_p
        self.raliBlurFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliBlend = self.lib.raliBlend
        self.raliBlend.restype = ctypes.c_void_p
        self.raliBlend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliBlendFixed = self.lib.raliBlendFixed
        self.raliBlendFixed.restype = ctypes.c_void_p
        self.raliBlendFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliWarpAffine = self.lib.raliWarpAffine
        self.raliWarpAffine.restype = ctypes.c_void_p
        self.raliWarpAffine.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.raliWarpAffineFixed = self.lib.raliWarpAffineFixed
        self.raliWarpAffineFixed.restype = ctypes.c_void_p
        self.raliWarpAffineFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool, ctypes.c_uint, ctypes.c_uint]

        self.raliFishEye = self.lib.raliFishEye
        self.raliFishEye.restype = ctypes.c_void_p
        self.raliFishEye.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]

        self.raliVignette = self.lib.raliVignette
        self.raliVignette.restype = ctypes.c_void_p
        self.raliVignette.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliVignetteFixed = self.lib.raliVignetteFixed
        self.raliVignetteFixed.restype = ctypes.c_void_p
        self.raliVignetteFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliJitter = self.lib.raliJitter
        self.raliJitter.restype = ctypes.c_void_p
        self.raliJitter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliJitterFixed = self.lib.raliJitterFixed
        self.raliJitterFixed.restype = ctypes.c_void_p
        self.raliJitterFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]

        self.raliSnPNoise = self.lib.raliSnPNoise
        self.raliSnPNoise.restype = ctypes.c_void_p
        self.raliSnPNoise.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliSnPNoiseFixed = self.lib.raliSnPNoiseFixed
        self.raliSnPNoiseFixed.restype = ctypes.c_void_p
        self.raliSnPNoiseFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliSnow = self.lib.raliSnow
        self.raliSnow.restype = ctypes.c_void_p
        self.raliSnow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliSnowFixed = self.lib.raliSnowFixed
        self.raliSnowFixed.restype = ctypes.c_void_p
        self.raliSnowFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliRain = self.lib.raliRain
        self.raliRain.restype = ctypes.c_void_p
        self.raliRain.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.raliRainFixed = self.lib.raliRainFixed
        self.raliRainFixed.restype = ctypes.c_void_p
        self.raliRainFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float, ctypes.c_bool]

        self.raliColorTemp = self.lib.raliColorTemp
        self.raliColorTemp.restype = ctypes.c_void_p
        self.raliColorTemp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliColorTempFixed = self.lib.raliColorTempFixed
        self.raliColorTempFixed.restype = ctypes.c_void_p
        self.raliColorTempFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]

        self.raliFog = self.lib.raliFog
        self.raliFog.restype = ctypes.c_void_p
        self.raliFog.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliFogFixed = self.lib.raliFogFixed
        self.raliFogFixed.restype = ctypes.c_void_p
        self.raliFogFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliLensCorrection = self.lib.raliLensCorrection
        self.raliLensCorrection.restype = ctypes.c_void_p
        self.raliLensCorrection.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p]

        self.raliLensCorrectionFixed = self.lib.raliLensCorrectionFixed
        self.raliLensCorrectionFixed.restype = ctypes.c_void_p
        self.raliLensCorrectionFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_bool]

        self.raliPixelate = self.lib.raliPixelate
        self.raliPixelate.restype = ctypes.c_void_p
        self.raliPixelate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]

        self.raliExposure = self.lib.raliExposure
        self.raliExposure.restype = ctypes.c_void_p
        self.raliExposure.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]

        self.raliExposureFixed = self.lib.raliExposureFixed
        self.raliExposureFixed.restype = ctypes.c_void_p
        self.raliExposureFixed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]

        self.raliCopy = self.lib.raliCopy
        self.raliCopy.restype = ctypes.c_void_p
        self.raliCopy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]

        self.raliNop = self.lib.raliNop
        self.raliNop.restype = ctypes.c_void_p
        self.raliNop.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]

        """ rali_api_info.h """
        self.raliGetOutputWidth = self.lib.raliGetOutputWidth
        self.raliGetOutputWidth.restype = ctypes.c_int
        self.raliGetOutputWidth.argtypes = [ctypes.c_void_p]

        self.raliGetOutputHeight = self.lib.raliGetOutputHeight
        self.raliGetOutputHeight.restype = ctypes.c_int
        self.raliGetOutputHeight.argtypes = [ctypes.c_void_p]

        self.raliGetOutputColorFormat = self.lib.raliGetOutputColorFormat
        self.raliGetOutputColorFormat.restype = ctypes.c_int
        self.raliGetOutputColorFormat.argtypes = [ctypes.c_void_p]

        self.remainingImagesCount = self.lib.raliGetRemainingImages
        self.remainingImagesCount.restype = ctypes.c_int
        self.remainingImagesCount.argtypes = [ctypes.c_void_p]

        self.raliSetSeed = self.lib.raliSetSeed
        self.raliSetSeed.restype = ctypes.c_void_p
        self.raliSetSeed.argtypes = [ctypes.c_ulonglong]

        self.raliGetSeed = self.lib.raliGetSeed
        self.raliGetSeed.restype = ctypes.c_ulonglong
        self.raliGetSeed.argtypes = []

        self.raliGetAugmentationBranchCount = self.lib.raliGetAugmentationBranchCount
        self.raliGetAugmentationBranchCount.restype = ctypes.c_int
        self.raliGetAugmentationBranchCount.argtypes = [ctypes.c_void_p]

        self.raliIsEmpty = self.lib.raliIsEmpty
        self.raliIsEmpty.restype = ctypes.c_int
        self.raliIsEmpty.argtypes = [ctypes.c_void_p]

        self.raliGetImageName = self.lib.raliGetImageName
        self.raliGetImageName.restype = ctypes.c_void_p
        self.raliGetImageName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]

        self.raliGetImageNameLen = self.lib.raliGetImageNameLen
        self.raliGetImageNameLen.restype = ctypes.c_uint
        self.raliGetImageNameLen.argtypes = [ctypes.c_void_p, ctypes.c_uint]

        """ rali_api_data_transfer.h """
        self.copyToOutput = self.lib.raliCopyToOutput
        self.copyToOutput.restype = ctypes.c_int
        self.copyToOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS")]

        self.copyToOutputTensor16 = self.lib.raliCopyToOutputTensor16
        self.copyToOutputTensor16.restype = ctypes.c_int
        self.copyToOutputTensor16.argtypes = [ctypes.c_void_p, ndpointer(dtype=np.float16, flags="C_CONTIGUOUS"), ctypes.c_uint, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint]

        self.copyToOutputTensor32 = self.lib.raliCopyToOutputTensor32
        self.copyToOutputTensor32.restype = ctypes.c_int
        self.copyToOutputTensor32.argtypes = [ctypes.c_void_p, ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"), ctypes.c_uint, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint]

        """ rali_api_meta_data.h"""
        self.raliCreateTextFileBasedLabelReader = self.lib.raliCreateTextFileBasedLabelReader
        self.raliCreateTextFileBasedLabelReader.restype = ctypes.c_void_p
        self.raliCreateTextFileBasedLabelReader.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.raliGetImageLabels = self.lib.raliGetImageLabels
        self.raliGetImageLabels.restype = ctypes.c_void_p
        self.raliGetImageLabels.argtypes = [ctypes.c_void_p, ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]

        self.raliCreateCifar10LabelReader = self.lib.raliCreateTextCifar10LabelReader
        self.raliCreateCifar10LabelReader.restype = ctypes.c_void_p
        self.raliCreateCifar10LabelReader.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
