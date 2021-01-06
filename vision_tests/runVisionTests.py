# Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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

from datetime import datetime
from subprocess import Popen, PIPE
import argparse
import os
import shutil, sys

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018-2020, AMD MIVision Generate Full Report"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiriti.NageshGowda@amd.com"
__status__ = "Shipping"


def shell(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output = p.communicate()[0][0:-1].decode("utf-8")
    return output


def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


# Vision Accuracy Tests
visionTestConfig = [
    '01_absDiff.gdf',
    '05_add.gdf',
    '06_and_u1.gdf',
    '06_and_u8.gdf',
    '07_box.gdf',
    '08_canny.gdf',
    '09_iyuv_channelCombine.gdf',
    '09_nv12_channelCombine.gdf',
    '09_rgb_channelCombine.gdf',
    '09_rgbx_channelCombine.gdf',
    '09_uyvy_channelCombine.gdf',
    '09_yuyv_channelCombine.gdf',
    '10_iyuv_channelExtract.gdf',
    '10_rgb_channelExtract.gdf',
    '10_rgbx_channelExtract.gdf',
    '11_colorConvert.gdf',
    '11_iyuv_colorConvert.gdf',
    '11_nv12_colorConvert.gdf',
    '11_nv21_colorConvert.gdf',
    '11_rgb_colorConvert.gdf',
    '11_rgbx_colorConvert.gdf',
    '11_uyvy_colorConvert.gdf',
    '11_yuyv_colorConvert.gdf',
    '12_convertDepth.gdf',
    '13_convolve.gdf',
    '14_dilate.gdf',
    '16_erode.gdf',
    '17_fastCorners.gdf',
    '18_gaussian.gdf',
    '20_halfScaleGaussian.gdf',
    '23_magnitude.gdf',
    '25_median.gdf',
    '27_multiply.gdf',
    '28_not.gdf',
    '30_or_u8.gdf',
    '30_or_u1.gdf',
    '31_phase.gdf',
    '34_scaleImage.gdf',
    '35_sobel.gdf',
    '36_subtract.gdf',
    '37_tableLookup.gdf',
    '38_threshold.gdf',
    '39_warpaffine.gdf',
    '40_warpperspective.gdf',
    '41_xor_u1.gdf',
    '41_xor_u8.gdf'
]

# OpenVX 1.0.1 Vision Functions 1080P
openvxNodes = [
#   Arithmetic kernels
    ('absdiff-1080p-u8', 'org.khronos.openvx.absdiff image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('absdiff-1080p-s16', 'org.khronos.openvx.absdiff image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
    ('add-1080p-u8-wrap', 'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,U008'),
    ('add-1080p-u8-sat', 'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !SATURATE image:1920,1080,U008'),
    ('add-1080p-s16-wrap', 'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('add-1080p-s16-wrap', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('add-1080p-s16-sat', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,U008 !SATURATE image:1920,1080,S016'),
    ('add-1080p-s16-wrap', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,S016 !WRAP image:1920,1080,S016'),
    ('add-1080p-s16-sat', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,S016 !SATURATE image:1920,1080,S016'),
    ('subtract-1080p-u8-wrap',
     'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,U008'),
    ('subtract-1080p-u8-sat',
     'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !SATURATE image:1920,1080,U008'),
    ('subtract-1080p-s16-wrap',
     'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('subtract-1080p-s16-wrap',
     'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('subtract-1080p-s16-sat',
     'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,U008 !SATURATE image:1920,1080,S016'),
    ('subtract-1080p-s16-wrap',
     'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,S016 !WRAP image:1920,1080,S016'),
    ('subtract-1080p-s16-sat',
     'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,S016 !SATURATE image:1920,1080,S016'),
    ('multiply-1080p-u8-wrap-round', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
    ('multiply-1080p-u8-wrap-trunc', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
    ('multiply-1080p-u8-sat-round', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
    ('multiply-1080p-u8-sat-trunc', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
    ('multiply-1080p-s16-wrap-round', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('multiply-1080p-s16-wrap-trunc', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('multiply-1080p-s16-sat-round', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('multiply-1080p-s16-sat-trunc', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('multiply-1080p-s16-wrap-round', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('multiply-1080p-s16-wrap-trunc', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('multiply-1080p-s16-sat-round', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('multiply-1080p-s16-sat-trunc', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('multiply-1080p-s16-wrap-round', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('multiply-1080p-s16-wrap-trunc', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('multiply-1080p-s16-sat-round', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('multiply-1080p-s16-sat-trunc', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('magnitude-1080p-S16',
     'org.khronos.openvx.magnitude image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
    ('phase-1080p-S16',
     'org.khronos.openvx.phase image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,U008'),
    ('weightedAverage-1080p-u8',
    'org.khronos.openvx.weighted_average image:1920,1080,U008 scalar:FLOAT32,0.25 image:1920,1080,U008 image:1920,1080,U008'),
#-----------------------------------------------------------------------------------------------------------------------
#   logical kernels
    ('and-1080p-u8-u8u8', 'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('and-1080p-u8-u8u1', 'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U008'),
    ('and-1080p-u8-u1u8', 'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U008'),
    ('and-1080p-u8-u1u1', 'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U008'),
    ('and-1080p-u1-u8u8', 'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U001'),
    ('and-1080p-u1-u8u1', 'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U001'),
    ('and-1080p-u1-u1u8', 'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U001'),
    ('and-1080p-u1-u1u1', 'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U001'),
    ('or-1080p-u8-u8u8', 'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('or-1080p-u8-u8u1', 'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U008'),
    ('or-1080p-u8-u1u8', 'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U008'),
    ('or-1080p-u8-u1u1', 'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U008'),
    ('or-1080p-u1-u8u8', 'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U001'),
    ('or-1080p-u1-u8u1', 'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U001'),
    ('or-1080p-u1-u1u8', 'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U001'),
    ('or-1080p-u1-u1u1', 'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U001'),
    ('xor-1080p-u8-u8u8', 'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('xor-1080p-u8-u8u1', 'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U008'),
    ('xor-1080p-u8-u1u8', 'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U008'),
    ('xor-1080p-u8-u1u1', 'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U008'),
    ('xor-1080p-u1-u8u8', 'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U001'),
    ('xor-1080p-u1-u8u1', 'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U001'),
    ('xor-1080p-u1-u1u8', 'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U001'),
    ('xor-1080p-u1-u1u1', 'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U001'),
    ('not-1080p-u8-u8', 'org.khronos.openvx.not image:1920,1080,U008 image:1920,1080,U008'),
    ('not-1080p-u8-u1', 'org.khronos.openvx.not image:1920,1080,U008 image:1920,1080,U001'),
    ('not-1080p-u1-u8', 'org.khronos.openvx.not image:1920,1080,U001 image:1920,1080,U008'),
    ('not-1080p-u1-u1', 'org.khronos.openvx.not image:1920,1080,U001 image:1920,1080,U001'),
#-----------------------------------------------------------------------------------------------------------------------
# Statistical kernels
    ('threshold-1080p-u8-binary',
     'org.khronos.openvx.threshold image:1920,1080,U008 threshold:BINARY,UINT8:INIT,127 image:1920,1080,U008'),
    ('threshold-1080p-u8-range',
     'org.khronos.openvx.threshold image:1920,1080,U008 threshold:RANGE,UINT8:INIT,100,200 image:1920,1080,U008'),
#-----------------------------------------------------------------------------------------------------------------------
# color_kernels
    ('convert_depth-1080p-s16-wrap',
    'org.khronos.openvx.convertdepth image:1920,1080,S016 image:1920,1080,U008 !WRAP scalar:INT32,1'),
    ('convert_depth-1080p-s16-sat',
    'org.khronos.openvx.convertdepth image:1920,1080,S016 image:1920,1080,U008 !SATURATE scalar:INT32,1'),
    ('convert_depth-1080p-u8',
    'org.khronos.openvx.convertdepth image:1920,1080,U008 image:1920,1080,S016 !WRAP scalar:INT32,1'),
    ('channel_extract-1080p-yuyv-pos0',
     'org.khronos.openvx.channel_extract image:1920,1080,YUYV !CHANNEL_Y image:1920,1080,U008'),
    ('channel_extract-1080p-uyvy-pos1',
     'org.khronos.openvx.channel_extract image:1920,1080,YUYV !CHANNEL_Y image:1920,1080,U008'),
    ('channel_extract-1080p-rgb-pos0',
     'org.khronos.openvx.channel_extract image:1920,1080,RGB2 !CHANNEL_R image:1920,1080,U008'),
    ('channel_extract-1080p-rgb-pos1',
     'org.khronos.openvx.channel_extract image:1920,1080,RGB2 !CHANNEL_G image:1920,1080,U008'),
    ('channel_extract-1080p-rgb-pos2',
     'org.khronos.openvx.channel_extract image:1920,1080,RGB2 !CHANNEL_B image:1920,1080,U008'),
    ('channel_extract-1080p-uyvy-pos0',
     'org.khronos.openvx.channel_extract image:1920,1080,UYVY !CHANNEL_U image:960,1080,U008'),
    ('channel_extract-1080p-yuyv-pos1',
     'org.khronos.openvx.channel_extract image:1920,1080,YUYV !CHANNEL_U image:960,1080,U008'),
    ('channel_extract-1080p-uyvy-pos2',
     'org.khronos.openvx.channel_extract image:1920,1080,UYVY !CHANNEL_V image:960,1080,U008'),
    ('channel_extract-1080p-uyvy-pos3',
     'org.khronos.openvx.channel_extract image:1920,1080,YUYV !CHANNEL_V image:960,1080,U008'),
    ('channel_combine-1080p-u8-nv12',
     'org.khronos.openvx.channel_combine image:1920,1080,U008 image:960,540,U008 image:960,540,U008 null image:1920,1080,NV12'),
    ('channel_combine-1080p-u8-uyvy',
     'org.khronos.openvx.channel_combine image:1920,1080,U008 image:960,1080,U008 image:960,1080,U008 null image:1920,1080,UYVY'),
    ('channel_combine-1080p-u8-yuyv',
     'org.khronos.openvx.channel_combine image:1920,1080,U008 image:960,1080,U008 image:960,1080,U008 null image:1920,1080,YUYV'),
    ('channel_combine-1080p-u8-rgb',
     'org.khronos.openvx.channel_combine image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 null image:1920,1080,RGB2'),
    ('channel_combine-1080p-u8-rgbx',
     'org.khronos.openvx.channel_combine image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,RGBA'),
    ('color_convert-1080p-rgbx-rgb',
     'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,RGB2'),
    ('color_convert-1080p-uyvy-rgb',
     'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,RGB2'),
    ('color_convert-1080p-yuyv-rgb',
     'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,RGB2'),
    ('color_convert-1080p-iyuv-rgb',
     'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,RGB2'),
    ('color_convert-1080p-nv12-rgb',
     'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,RGB2'),
    ('color_convert-1080p-nv21-rgb',
     'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,RGB2'),
    ('color_convert-1080p-rgb-rgbx',
     'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,RGBA'),
    ('color_convert-1080p-uyvy-rgbx',
     'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,RGBA'),
    ('color_convert-1080p-yuyv-rgbx',
     'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,RGBA'),
    ('color_convert-1080p-iyuv-rgbx',
     'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,RGBA'),
    ('color_convert-1080p-nv12-rgbx',
     'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,RGBA'),
    ('color_convert-1080p-nv21-rgbx',
     'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,RGBA'),
    ('color_convert-1080p-rgb-iyuv',
     'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,IYUV'),
    ('color_convert-1080p-rgbx-iyuv',
     'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,IYUV'),
    ('color_convert-1080p-uyvy-iyuv',
     'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,IYUV'),
    ('color_convert-1080p-yuyv-iyuv',
     'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,IYUV'),
    ('color_convert-1080p-rgb-nv12',
     'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,NV12'),
    ('color_convert-1080p-rgbx-nv12',
     'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,NV12'),
    ('color_convert-1080p-uyvy-nv12',
     'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,NV12'),
    ('color_convert-1080p-yuyv-nv12',
     'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,NV12'),
    ('color_convert-1080p-nv12-iyuv',
     'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,IYUV'),
    ('color_convert-1080p-nv21-iyuv',
     'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,IYUV'),
    ('color_convert-1080p-iyuv-nv12',
     'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,NV12'),
    ('color_convert-1080p-rgb-yuv4',
    'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,YUV4'),  
    ('color_convert-1080p-rgbx-yuv4',
    'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,YUV4'),  
    ('color_convert-1080p-nv12-yuv4',
    'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,YUV4'),  
    ('color_convert-1080p-nv21-yuv4',
    'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,YUV4'),  
    ('color_convert-1080p-iyuv-yuv4',
    'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,YUV4'),      
#-----------------------------------------------------------------------------------------------------------------------
# filter_kernels
    ('box_3x3-1080p-u8',
     'org.khronos.openvx.box_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('dilate_3x3-1080p-u8-u8',
     'org.khronos.openvx.dilate_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('dilate_3x3-1080p-u8-u1',
     'org.khronos.openvx.dilate_3x3 image:1920,1080,U008 image:1920,1080,U001'),
    ('dilate_3x3-1080p-u1-u8',
     'org.khronos.openvx.dilate_3x3 image:1920,1080,U001 image:1920,1080,U008'),
    ('dilate_3x3-1080p-u1-u1',
     'org.khronos.openvx.dilate_3x3 image:1920,1080,U001 image:1920,1080,U001'),
    ('erode_3x3-1080p-u8',
     'org.khronos.openvx.erode_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('erode_3x3-1080p-u8-u1',
     'org.khronos.openvx.erode_3x3 image:1920,1080,U008 image:1920,1080,U001'),
    ('erode_3x3-1080p-u1-u8',
     'org.khronos.openvx.erode_3x3 image:1920,1080,U001 image:1920,1080,U008'),
    ('erode_3x3-1080p-u1-u1',
     'org.khronos.openvx.erode_3x3 image:1920,1080,U001 image:1920,1080,U001'),
    ('median_3x3-1080p-u8',
     'org.khronos.openvx.median_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('gaussian_3x3-1080p-u8',
     'org.khronos.openvx.gaussian_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('halfscale_gaussian_3x3-1080p-u8',
     'org.khronos.openvx.halfscale_gaussian image:1920,1080,U008 image:1920,1080,U008 scalar:INT32,3'),
    ('halfscale_gaussian_5x5-1080p-u8',
     'org.khronos.openvx.halfscale_gaussian image:1920,1080,U008 image:1920,1080,U008 scalar:INT32,5'),
    ('convolve-i080p-u8',
     'org.khronos.openvx.custom_convolution image:1920,1080,U008 convolution:3,3 image:1920,1080,U008'),
    ('convolve-i080p-s16',
     'org.khronos.openvx.custom_convolution image:1920,1080,U008 convolution:3,3 image:1920,1080,S016'),
    ('sobel_3x3-1080p-u8-gxgy',
     'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 image:1920,1080,S016 image:1920,1080,S016'),
    ('sobel_3x3-1080p-u8-gx',
     'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 image:1920,1080,S016 null'),
    ('sobel_3x3-1080p-u8-gy',
     'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 null image:1920,1080,S016'),
#-----------------------------------------------------------------------------------------------------------------------
# geometric_kernels
    ('scale_image-1080p-u8-nearest',
     'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !NEAREST_NEIGHBOR'),
    ('scale_image-1080p-u8-bilinear',
     'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR'),
    ('scale_image-1080p-u8-bilinear-replicate',
     'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR attr:BORDER_MODE:REPLICATE'),
    ('scale_image-1080p-u8-bilinear-constant',
     'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR attr:BORDER_MODE:CONSTANT,0'),
    ('scale_image-1080p-u8-area',
     'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !AREA'),
    ('warp_affine-1080p-u8-nearest',
     'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !NEAREST_NEIGHBOR image:1920,1080,U008'),
    ('warp_affine-1080p-u8-nearest-constant',
     'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('warp_affine-1080p-u8-bilinear',
     'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !BILINEAR image:1920,1080,U008'),
    ('warp_affine-1080p-u8-bilinear-constant',
     'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('warp_perspective-1080p-u8-nearest',
     'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !NEAREST_NEIGHBOR image:1920,1080,U008'),
    ('warp_perspective-1080p-u8-nearest-constant',
     'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('warp_perspective-1080p-u8-bilinear',
     'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !BILINEAR image:1920,1080,U008'),
    ('warp_perspective-1080p-u8-bilinear-constant',
     'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('remap-1080p-S16-nearest',
    'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008'),
    ('remap-1080p-S16-nearest-constant',
    'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('remap-1080p-S16-bilinear',
    'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !BILINEAR image:1920,1080,U008'),
    ('remap-1080p-S16-bilinear-constant',
    'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
#-----------------------------------------------------------------------------------------------------------------------
# vision_kernels
    ('fast_corners-1080p-u8-nosuppression',
     'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,1 array:KEYPOINT,1000 scalar:SIZE,0'),
    ('fast_corners-1080p-u8-suppression',
     'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,0 array:KEYPOINT,1000 scalar:SIZE,0'),
    ('table_lookup-1080p-u8',
     'org.khronos.openvx.table_lookup image:1920,1080,U008 lut:UINT8,256 image:1920,1080,U008')
#-----------------------------------------------------------------------------------------------------------------------
]
# ('remap-1080p-S16-nearest',
#  'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008'),
# ('remap-1080p-S16-bilinear',
#  'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !BILINEAR image:1920,1080,U008'),


# openvxNodes = [
#     ('absdiff-1080p-u8', 'org.khronos.openvx.absdiff image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
#     ('absdiff-1080p-s16', 'org.khronos.openvx.absdiff image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
#     ('accumulate-1080p-u8',
#      'org.khronos.openvx.accumulate image:1920,1080,U008 image:1920,1080,S016'),
#     ('accumulate_square-1080p-u8',
#      'org.khronos.openvx.accumulate_square image:1920,1080,U008 scalar:UINT32,0 image:1920,1080,S016'),
#     ('accumulate_weighted-1080p-u8',
#      'org.khronos.openvx.accumulate_weighted image:1920,1080,U008 scalar:FLOAT32,0.3 image:1920,1080,U008'),
#     ('add-1080p-u8', 'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,U008'),
#     ('add-1080p-u8', 'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !SATURATE image:1920,1080,U008'),
#     ('add-1080p-s16', 'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
#     ('add-1080p-s16', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
#     ('add-1080p-s16', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,U008 !SATURATE image:1920,1080,S016'),
#     ('add-1080p-s16', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,S016 !WRAP image:1920,1080,S016'),
#     ('add-1080p-s16', 'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,S016 !SATURATE image:1920,1080,S016'),
#     ('and-1080p-u8', 'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
#     ('box_3x3-1080p-u8',
#      'org.khronos.openvx.box_3x3 image:1920,1080,U008 image:1920,1080,U008'),
#     ('canny_edge_detector-1080p-u8',
#      'org.khronos.openvx.canny_edge_detector image:1920,1080,U008 threshold:RANGE,UINT8:INIT,80,100 scalar:INT32,3 !NORM_L1 image:1920,1080,U008'),
#     ('channel_combine-1080p-u8',
#      'org.khronos.openvx.channel_combine image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,RGB2'),
#     ('dilate_3x3-1080p-u8',
#      'org.khronos.openvx.dilate_3x3 image:1920,1080,U008 image:1920,1080,U008'),
#     ('erode_3x3-1080p-u8',
#      'org.khronos.openvx.erode_3x3 image:1920,1080,U008 image:1920,1080,U008'),
#     ('fast_corners-1080p-u8', 'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,1 array:KEYPOINT,1000 scalar:UINT32,0'),
#     ('gaussian_3x3-1080p-u8',
#      'org.khronos.openvx.gaussian_3x3 image:1920,1080,U008 image:1920,1080,U008'),
#     ('gaussian_pyramid-1080p-u8',
#      'org.khronos.openvx.gaussian_pyramid image:1920,1080,U008 pyramid:4,HALF,1920,1080,U008'),
#     ('halfscale_gaussian-1080p-u8',
#      'org.khronos.openvx.halfscale_gaussian image:1920,1080,U008 image:960,540,U008 scalar:INT32,5'),
#     ('harris_corners-1080p-u8', 'org.khronos.openvx.harris_corners image:1920,1080,U008 scalar:FLOAT32,0.00001 scalar:FLOAT32,20.0 scalar:FLOAT32,0.10 scalar:INT32,3 scalar:INT32,5 array:KEYPOINT,1000 scalar:UINT32,0'),
#     ('histogram-1080p-u8',
#      'org.khronos.openvx.histogram image:1920,1080,U008 distribution:2,0,256'),
#     ('integral_image-1080p-u8',
#      'org.khronos.openvx.integral_image image:1920,1080,U008 image:1920,1080,U032'),
#     ('magnitude-1080p-S16',
#      'org.khronos.openvx.magnitude image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
#     ('mean_stddev-1080p-u8',
#      'org.khronos.openvx.mean_stddev image:1920,1080,U008 scalar:FLOAT32,0.0 scalar:FLOAT32,0.0'),
#     ('median_3x3-1080p-u8',
#      'org.khronos.openvx.median_3x3 image:1920,1080,U008 image:1920,1080,U008'),
#     ('minmaxloc-1080p-u8',
#      'org.khronos.openvx.minmaxloc image:1920,1080,U008 scalar:UINT8,0 scalar:UINT8,0 array:COORDINATES2D,1000 array:COORDINATES2D,1000 scalar:UINT32,0 scalar:UINT32,0'),
#     ('multiply-1080p-u8', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
#     ('multiply-1080p-u8', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
#     ('multiply-1080p-u8', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
#     ('multiply-1080p-u8', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
#     ('multiply-1080p-s16', 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
#     ('not-1080p-u8', 'org.khronos.openvx.not image:1920,1080,U008 image:1920,1080,U008'),
#     ('optical_flow_pyr_lk-1080p-u8',
#      'org.khronos.openvx.optical_flow_pyr_lk pyramid:4,HALF,1920,1080,U008 pyramid:4,HALF,1920,1080,U008 array:KEYPOINT,1000 array:KEYPOINT,1000 array:KEYPOINT,1000 scalar:ENUM,CRITERIA_BOTH scalar:FLOAT32,0.1 scalar:UINT32,4 scalar:BOOL,0 scalar:SIZE,5'),
#     ('or-1080p-u8', 'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
#     ('phase-1080p-S16',
#      'org.khronos.openvx.phase image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,U008'),
#     ('remap-1080p-S16',
#      'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008 !BORDER_MODE_CONSTANT'),
#     ('scale_image-1080p-u8',
#      'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR'),
#     ('sobel_3x3-1080p-u8',
#      'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 image:1920,1080,S016 image:1920,1080,S016'),
#     ('subtract-1080p-u8',
#      'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,U008'),
#     ('subtract-1080p-u8',
#      'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !SATURATE image:1920,1080,U008'),
#     ('subtract-1080p-s16',
#      'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
#     ('subtract-1080p-s16',
#      'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
#     ('subtract-1080p-s16',
#      'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,U008 !SATURATE image:1920,1080,S016'),
#     ('subtract-1080p-s16',
#      'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,S016 !WRAP image:1920,1080,S016'),
#     ('subtract-1080p-s16',
#      'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,S016 !SATURATE image:1920,1080,S016'),
#     ('table_lookup-1080p-u8',
#      'org.khronos.openvx.table_lookup image:1920,1080,U008 lut:UINT8,256 image:1920,1080,U008'),
#     ('threshold-1080p-u8',
#      'org.khronos.openvx.threshold image:1920,1080,U008 threshold:BINARY,UINT8:INIT,127 image:1920,1080,U008'),
#     ('threshold-1080p-u8',
#      'org.khronos.openvx.threshold image:1920,1080,U008 threshold:RANGE,UINT8:INIT,100,200 image:1920,1080,U008'),
#     ('warp_affine-1080p-u8',
#      'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !BILINEAR image:1920,1080,U008'),
#     ('warp_perspective-1080p-u8',
#      'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !NEAREST_NEIGHBOR image:1920,1080,U008'),
#     ('xor-1080p-u8', 'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008')
# ]

#  Popular Video Sizes
# { 2160p: 3840x2160, 1440p: 2560x1440, 1080p: 1920x1080, 720p: 1280x720, 480p: 854x480, 360p: 640x360, 240p: 426x240 }
# openvxNodeTestConfig = [
#     # absdiff U8 - U8
#     ('absdiff-240p-u8', 'org.khronos.openvx.absdiff image:426,240,U008 image:426,240,U008 image:426,240,U008'),
#     ('absdiff-360p-u8', 'org.khronos.openvx.absdiff image:640,360,U008 image:640,360,U008 image:640,360,U008'),
#     ('absdiff-480p-u8', 'org.khronos.openvx.absdiff image:854,480,U008 image:854,480,U008 image:854,480,U008'),
#     ('absdiff-720p-u8', 'org.khronos.openvx.absdiff image:1280,720,U008 image:1280,720,U008 image:1280,720,U008'),
#     ('absdiff-1080p-u8', 'org.khronos.openvx.absdiff image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
#     ('absdiff-1440p-u8', 'org.khronos.openvx.absdiff image:2560,1440,U008 image:2560,1440,U008 image:2560,1440,U008'),
#     ('absdiff-2160p-u8', 'org.khronos.openvx.absdiff image:3840,2160,U008 image:3840,2160,U008 image:3840,2160,U008'),
#     # absdiff S16 - S16
#     ('absdiff-240p-s16', 'org.khronos.openvx.absdiff image:426,240,S016 image:426,240,S016 image:426,240,S016'),
#     ('absdiff-360p-s16', 'org.khronos.openvx.absdiff image:640,360,S016 image:640,360,S016 image:640,360,S016'),
#     ('absdiff-480p-s16', 'org.khronos.openvx.absdiff image:854,480,S016 image:854,480,S016 image:854,480,S016'),
#     ('absdiff-720p-s16', 'org.khronos.openvx.absdiff image:1280,720,S016 image:1280,720,S016 image:1280,720,S016'),
#     ('absdiff-1080p-s16', 'org.khronos.openvx.absdiff image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
#     ('absdiff-1440p-s16', 'org.khronos.openvx.absdiff image:2560,1440,S016 image:2560,1440,S016 image:2560,1440,S016'),
#     ('absdiff-2160p-s16', 'org.khronos.openvx.absdiff image:3840,2160,S016 image:3840,2160,S016 image:3840,2160,S016')
# ]

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--runvx_directory',    type=str, default='',
                    help='RunVX Executable directory - required')
parser.add_argument('--hardware_mode',      type=str, default='CPU',
                    help='OpenVX Vision Function target - optional (default:CPU [options:CPU/GPU])')
args = parser.parse_args()

runvxDir = args.runvx_directory
hardwareMode = args.hardware_mode

# check arguments
if runvxDir == '':
    print(
        "ERROR: RunVX Executable Directory Required")
    exit()
if hardwareMode not in ('CPU', 'GPU'):
    print("ERROR: OpenVX Hardware supported - CPU or GPU]")
    exit()

# RunVX Application
runVX_exe = runvxDir+'/runvx'
RunVXapp = os.path.abspath(runVX_exe)
scriptPath = os.path.dirname(os.path.realpath(__file__))
if(os.path.isfile(RunVXapp)):
    print("STATUS: RunVX path - "+RunVXapp)
else:
    print("\nERROR: RunVX Executable Not Found\n")
    exit()

# print("\nrunVisionTests - OpenVX Vision Tests V-"+__version__+"\n")
# os.system('(cd '+scriptPath+'/gdfs; mkdir -p openvx_test_results)')
# for i in range(len(visionTestConfig[:])):
#     testFileName = visionTestConfig[i]
#     print("Running Test Script: "+testFileName)
#     os.system(RunVXapp+' -frames:100 -affinity:' +
#                   hardwareMode+' -dump-profile file '+scriptPath+'/gdfs/'+testFileName+' | tee -a '+scriptPath+'/gdfs/openvx_test_results/VisionOutput.log')
#     print("\n")

# print("\nrunVisionTests - OpenVX Node Tests V-"+__version__+"\n")
# os.system('mkdir openvx_node_results')
# for i in range(len(openvxNodeTestConfig)):
#     nodeTestName, nodeTest = openvxNodeTestConfig[i]
#     print("Running OpenVX Node: "+nodeTestName)
#     os.system(RunVXapp+' -frames:1000 -affinity:'+hardwareMode+' -dump-profile node '+nodeTest+' | tee -a openvx_node_results/nodeTestOutput.log')
#     print("\n")

print("\nrunVisionTests - OpenVX Node Performance V-"+__version__+"\n")
outputDirectory = 'openvx_node_results'
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
else:
    shutil.rmtree(outputDirectory)
    os.makedirs(outputDirectory)
for i in range(len(openvxNodes[:])):
    nodeName, nodeFormat = openvxNodes[i]
    echo1 = 'Running OpenVX Node - '+nodeName
    os.system('echo '+echo1 +
              ' | tee -a openvx_node_results/nodePerformanceOutput.log')
    os.system(RunVXapp+' -frames:1000 -affinity:' +
              hardwareMode+' -dump-profile node '+nodeFormat+' | tee -a openvx_node_results/nodePerformanceOutput.log')
    print("\n")


orig_stdout = sys.stdout
sys.stdout = open('openvx_node_results/nodePerformance.md','a')
echo_1 = '| OpenVX Node                          |Frames Count| tmp (ms) | avg (ms) | min (ms) | max (ms) |'
print(echo_1)
echo_2 = '|--------------------------------------|------------|----------|----------|----------|----------|'
print(echo_2)
sys.stdout = orig_stdout
print(echo_1)
print(echo_2)
if hardwareMode == 'CPU':
    runAwk_csv = r'''awk 'BEGIN { node = "xxx"; } /Running OpenVX Node - / { node = $5; } /CPU,GRAPH/ { printf("| %-36s | %10d | %8.3f | %8.3f | %8.3f | %8.3f |\n", node, $1, $2, $3, $4, $5) }' openvx_node_results/nodePerformanceOutput.log | tee -a openvx_node_results/nodePerformance.md'''
elif hardwareMode == 'GPU':
    runAwk_csv = r'''awk 'BEGIN { node = "xxx"; } /Running OpenVX Node - / { node = $5; } /GPU,GRAPH/ { printf("| %-36s | %10d | %8.3f | %8.3f | %8.3f | %8.3f |\n", node, $1, $2, $3, $4, $5) }' openvx_node_results/nodePerformanceOutput.log | tee -a openvx_node_results/nodePerformance.md'''
os.system(runAwk_csv)

# get data
platform_name = shell('hostname')
platform_name_fq = shell('hostname --all-fqdns')
platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

file_dtstr = datetime.now().strftime("%Y%m%d")
report_filename = 'platform_report_%s_%s.md' % (platform_name, file_dtstr)
report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
sys_info = shell('inxi -c0 -S')

cpu_info = shell('inxi -c0 -C')
cpu_info = cpu_info.split('\n')[0]  # strip out clock speeds

gpu_info = shell('inxi -c0 -G')
gpu_info = gpu_info.split('\n')[0]  # strip out X info

memory_info = shell('inxi -c 0 -m')
board_info = shell('inxi -c0 -M')

# Write Report
with open(report_filename, 'w') as f:
    f.write("MIVisionX Lite - OpenVX 1.0.1 Report\n")
    f.write("================================\n")
    f.write("\n")

    f.write("Generated: %s\n" % report_dtstr)
    f.write("\n")

    f.write("Platform: %s (%s)\n" % (platform_name_fq, platform_ip))
    f.write("--------\n")
    f.write("\n")

    write_formatted(sys_info, f)
    write_formatted(cpu_info, f)
    write_formatted(gpu_info, f)
    write_formatted(board_info, f)
    write_formatted(memory_info, f)

    f.write("\n\nBenchmark Report\n")
    f.write("--------\n")
    f.write("Hardware: %s\n" % hardwareMode)
    f.write("\n")
    with open('openvx_node_results/nodePerformance.md') as benchmarkFile:
        for line in benchmarkFile:
            f.write("%s" % line)
    f.write("\n")

    f.write("\n\n---\nCopyright AMD ROCm MIVisionX 2018 - 2020 -- runVisionTests.py V-"+__version__+"\n")

print("\nrunVisionTests.py completed - V:"+__version__+"\n")