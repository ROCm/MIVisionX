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
# Arithmetic kernels
    ('AbsDiff_U8_U8U8',                           'org.khronos.openvx.absdiff image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('AbsDiff_S16_S16S16_Sat',                    'org.khronos.openvx.absdiff image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
    ('Add_U8_U8U8_Wrap',                          'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,U008'),
    ('Add_U8_U8U8_Sat',                           'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !SATURATE image:1920,1080,U008'),
    ('Add_S16_U8U8_Wrap',                         'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('Add_S16_S16U8_Wrap',                        'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('Add_S16_S16U8_Sat',                         'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,U008 !SATURATE image:1920,1080,S016'),
    ('Add_S16_S16S16_Wrap',                       'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,S016 !WRAP image:1920,1080,S016'),
    ('Add_S16_S16S16_Sat',                        'org.khronos.openvx.add image:1920,1080,S016 image:1920,1080,S016 !SATURATE image:1920,1080,S016'),
    ('Sub_U8_U8U8_Wrap',                          'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,U008'),
    ('Sub_U8_U8U8_Sat',                           'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !SATURATE image:1920,1080,U008'),
    ('Sub_S16_U8U8_Wrap',                         'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('Sub_S16_S16U8_Wrap',                        'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('Sub_S16_S16U8_Sat',                         'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,U008 !SATURATE image:1920,1080,S016'),
    ('Sub_S16_S16S16_Wrap',                       'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,S016 !WRAP image:1920,1080,S016'),
    ('Sub_S16_S16S16_Sat',                        'org.khronos.openvx.subtract image:1920,1080,S016 image:1920,1080,S016 !SATURATE image:1920,1080,S016'),
    ('Mul_U8_U8U8_Wrap_Trunc',                    'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
    ('Mul_U8_U8U8_Wrap_Round',                    'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
    ('Mul_U8_U8U8_Sat_Trunc',                     'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
    ('Mul_U8_U8U8_Sat_Round',                     'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
    ('Mul_S16_U8U8_Wrap_Trunc',                   'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_U8U8_Wrap_Round',                   'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_U8U8_Sat_Trunc',                    'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_U8U8_Sat_Round',                    'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16U8_Wrap_Trunc',                  'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16U8_Wrap_Round',                  'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16U8_Sat_Trunc',                   'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16U8_Sat_Round',                   'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,U008 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16S16_Wrap_Trunc',                 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16S16_Wrap_Round',                 'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16S16_Sat_Trunc',                  'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16S16_Sat_Round',                  'org.khronos.openvx.multiply image:1920,1080,S016 image:1920,1080,S016 scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Magnitude_S16_S16S16',                      'org.khronos.openvx.magnitude image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
    ('Phase_U8_S16S16',                           'org.khronos.openvx.phase image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,U008'),
    ('WeightedAverage_U8_U8U8',                   'org.khronos.openvx.weighted_average image:1920,1080,U008 scalar:FLOAT32,0.25 image:1920,1080,U008 image:1920,1080,U008'),
# logical kernels
    ('And_U8_U8U8',                               'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('And_U8_U8U1',                               'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U008'),
    ('And_U8_U1U8',                               'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U008'),
    ('And_U8_U1U1',                               'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U008'),
    ('And_U1_U8U8',                               'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U001'),
    ('And_U1_U8U1',                               'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U001'),
    ('And_U1_U1U8',                               'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U001'),
    ('And_U1_U1U1',                               'org.khronos.openvx.and image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U001'),
    ('Or_U8_U8U8',                                'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('Or_U8_U8U1',                                'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U008'),
    ('Or_U8_U1U8',                                'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U008'),
    ('Or_U8_U1U1',                                'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U008'),
    ('Or_U1_U8U8',                                'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U001'),
    ('Or_U1_U8U1',                                'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U001'),
    ('Or_U1_U1U8',                                'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U001'),
    ('Or_U1_U1U1',                                'org.khronos.openvx.or image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U001'),
    ('Xor_U8_U8U8',                               'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('Xor_U8_U8U1',                               'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U008'),
    ('Xor_U8_U1U8',                               'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U008'),
    ('Xor_U8_U1U1',                               'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U008'),
    ('Xor_U1_U8U8',                               'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U001'),
    ('Xor_U1_U8U1',                               'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U001 image:1920,1080,U001'),
    ('Xor_U1_U1U8',                               'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U008 image:1920,1080,U001'),
    ('Xor_U1_U1U1',                               'org.khronos.openvx.xor image:1920,1080,U001 image:1920,1080,U001 image:1920,1080,U001'),
    ('Not_U8_U8',                                 'org.khronos.openvx.not image:1920,1080,U008 image:1920,1080,U008'),
    ('Not_U1_U8',                                 'org.khronos.openvx.not image:1920,1080,U008 image:1920,1080,U001'),
    ('Not_U8_U1',                                 'org.khronos.openvx.not image:1920,1080,U001 image:1920,1080,U008'),
    ('Not_U1_U1',                                 'org.khronos.openvx.not image:1920,1080,U001 image:1920,1080,U001'),
# Statistical kernels
    ('Threshold_U8_U8_Binary',                    'org.khronos.openvx.threshold image:1920,1080,U008 threshold:BINARY,UINT8:INIT,127 image:1920,1080,U008'),
    ('Threshold_U8_U8_Range',                     'org.khronos.openvx.threshold image:1920,1080,U008 threshold:RANGE,UINT8:INIT,100,200 image:1920,1080,U008'),
# Color kernels
    ('ColorDepth_U8_S16_Wrap',                    'org.khronos.openvx.convertdepth image:1920,1080,S016 image:1920,1080,U008 !WRAP scalar:INT32,1'),
    ('ColorDepth_U8_S16_Sat',                     'org.khronos.openvx.convertdepth image:1920,1080,S016 image:1920,1080,U008 !SATURATE scalar:INT32,1'),
    ('ColorDepth_S16_U8',                         'org.khronos.openvx.convertdepth image:1920,1080,U008 image:1920,1080,S016 !WRAP scalar:INT32,1'),
    ('ChannelExtract_U8_U16_Pos0',                'org.khronos.openvx.channel_extract image:1920,1080,YUYV !CHANNEL_Y image:1920,1080,U008'),
    ('ChannelExtract_U8_U24_Pos0',                'org.khronos.openvx.channel_extract image:1920,1080,RGB2 !CHANNEL_R image:1920,1080,U008'),
    ('ChannelExtract_U8_U24_Pos1',                'org.khronos.openvx.channel_extract image:1920,1080,RGB2 !CHANNEL_G image:1920,1080,U008'),
    ('ChannelExtract_U8_U24_Pos2',                'org.khronos.openvx.channel_extract image:1920,1080,RGB2 !CHANNEL_B image:1920,1080,U008'),
    ('ChannelExtract_U8_U32_Pos0_UYVY',           'org.khronos.openvx.channel_extract image:1920,1080,UYVY !CHANNEL_U image:960,1080,U008'),
    ('ChannelExtract_U8_U32_Pos1_YUYV',           'org.khronos.openvx.channel_extract image:1920,1080,YUYV !CHANNEL_U image:960,1080,U008'),
    ('ChannelExtract_U8_U32_Pos2_UYVY',           'org.khronos.openvx.channel_extract image:1920,1080,UYVY !CHANNEL_V image:960,1080,U008'),
    ('ChannelExtract_U8_U32_Pos3_YUYV',           'org.khronos.openvx.channel_extract image:1920,1080,YUYV !CHANNEL_V image:960,1080,U008'),
    #('ChannelCombine_U16_U8U8',                  'org.khronos.openvx.channel_combine image:1920,1080,U008 image:960,540,U008 image:960,540,U008 null image:1920,1080,NV12'),
    ('ChannelCombine_U32_U8U8U8_UYVY',            'org.khronos.openvx.channel_combine image:1920,1080,U008 image:960,1080,U008 image:960,1080,U008 null image:1920,1080,UYVY'),
    ('ChannelCombine_U32_U8U8U8_YUYV',            'org.khronos.openvx.channel_combine image:1920,1080,U008 image:960,1080,U008 image:960,1080,U008 null image:1920,1080,YUYV'),
    ('ChannelCombine_U24_U8U8U8_RGB',             'org.khronos.openvx.channel_combine image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 null image:1920,1080,RGB2'),
    ('ChannelCombine_U32_U8U8U8U8_RGBX',          'org.khronos.openvx.channel_combine image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,RGBA'),
    ('ColorConvert_RGB_RGBX',                     'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,RGB2'),
    ('ColorConvert_RGB_UYVY',                     'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,RGB2'),
    ('ColorConvert_RGB_YUYV',                     'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,RGB2'),
    ('ColorConvert_RGB_IYUV',                     'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,RGB2'),
    ('ColorConvert_RGB_NV12',                     'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,RGB2'),
    ('ColorConvert_RGB_NV21',                     'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,RGB2'),
    ('ColorConvert_RGBX_RGB',                     'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_UYVY',                    'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_YUYV',                    'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_IYUV',                    'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_NV12',                    'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_NV21',                    'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,RGBA'),
    ('ColorConvert_IYUV_RGB',                     'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,IYUV'),
    ('ColorConvert_IYUV_RGBX',                    'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,IYUV'),
    ('FormatConvert_IYUV_UYVY',                   'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,IYUV'),
    ('FormatConvert_IYUV_YUYV',                   'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,IYUV'),
    ('ColorConvert_NV12_RGB',                     'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,NV12'),
    ('ColorConvert_NV12_RGBX',                    'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,NV12'),
    ('FormatConvert_NV12_UYVY',                   'org.khronos.openvx.color_convert image:1920,1080,UYVY image:1920,1080,NV12'),
    ('FormatConvert_NV12_YUYV',                   'org.khronos.openvx.color_convert image:1920,1080,YUYV image:1920,1080,NV12'),
    #('FormatConvert_IUV_UV12',                   'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,IYUV'),
    #('FormatConvert_IUV_UV12',                   'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,IYUV'),
    #('FormatConvert_UV12_IUV',                   'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,NV12'),
    ('ColorConvert_YUV4_RGB',                     'org.khronos.openvx.color_convert image:1920,1080,RGB2 image:1920,1080,YUV4'),  
    ('ColorConvert_YUV4_RGBX',                    'org.khronos.openvx.color_convert image:1920,1080,RGBA image:1920,1080,YUV4'),  
    #('FormatConvert_UV_UV12',                    'org.khronos.openvx.color_convert image:1920,1080,NV12 image:1920,1080,YUV4'),  
    #('FormatConvert_UV_UV12',                    'org.khronos.openvx.color_convert image:1920,1080,NV21 image:1920,1080,YUV4'),  
    #('ScaleUp2x2_U8_U8',                         'org.khronos.openvx.color_convert image:1920,1080,IYUV image:1920,1080,YUV4'),      
# filter kernels
    ('Box_U8_U8_3x3',                             'org.khronos.openvx.box_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('Dilate_U8_U8_3x3',                          'org.khronos.openvx.dilate_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('Dilate_U1_U8_3x3',                          'org.khronos.openvx.dilate_3x3 image:1920,1080,U008 image:1920,1080,U001'),
    ('Dilate_U8_U1_3x3',                          'org.khronos.openvx.dilate_3x3 image:1920,1080,U001 image:1920,1080,U008'),
    ('Dilate_U1_U1_3x3',                          'org.khronos.openvx.dilate_3x3 image:1920,1080,U001 image:1920,1080,U001'),
    ('Erode_U8_U8_3x3',                           'org.khronos.openvx.erode_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('Erode_U1_U8_3x3',                           'org.khronos.openvx.erode_3x3 image:1920,1080,U008 image:1920,1080,U001'),
    ('Erode_U8_U1_3x3',                           'org.khronos.openvx.erode_3x3 image:1920,1080,U001 image:1920,1080,U008'),
    ('Erode_U1_U1_3x3',                           'org.khronos.openvx.erode_3x3 image:1920,1080,U001 image:1920,1080,U001'),
    ('Median_U8_U8_3x3',                          'org.khronos.openvx.median_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('Gaussian_U8_U8_3x3',                        'org.khronos.openvx.gaussian_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('HalfGaussian_U8_U8_3x3',                    'org.khronos.openvx.halfscale_gaussian image:1920,1080,U008 image:1920,1080,U008 scalar:INT32,3'),
    ('HalfGaussian_U8_U8_5x5',                    'org.khronos.openvx.halfscale_gaussian image:1920,1080,U008 image:1920,1080,U008 scalar:INT32,5'),
    ('Convolve_U8_U8_3x3',                        'org.khronos.openvx.custom_convolution image:1920,1080,U008 convolution:3,3 image:1920,1080,U008'),
    ('Convolve_S16_U8_3x3',                       'org.khronos.openvx.custom_convolution image:1920,1080,U008 convolution:3,3 image:1920,1080,S016'),
    ('Sobel_S16S16_U8_3x3_GXY',                   'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 image:1920,1080,S016 image:1920,1080,S016'),
    ('Sobel_S16_U8_3x3_GX',                       'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 image:1920,1080,S016 null'),
    ('Sobel_S16_U8_3x3_GY',                       'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 null image:1920,1080,S016'),
# geometric kernels
    ('ScaleImage_U8_U8_Nearest',                   'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !NEAREST_NEIGHBOR'),
    ('ScaleImage_U8_U8_bilinear',                  'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR'),
    ('ScaleImage_U8_U8_bilinear_replicate',        'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR attr:BORDER_MODE:REPLICATE'),
    ('ScaleImage_U8_U8_bilinear_constant',         'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR attr:BORDER_MODE:CONSTANT,0'),
    ('ScaleImage_U8_U8_u8_area',                   'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !AREA'),
    ('WarpAffine_U8_U8_Nearest',                   'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !NEAREST_NEIGHBOR image:1920,1080,U008'),
    ('WarpAffine_U8_U8_Nearest_constant',          'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('WarpAffine_U8_U8_Bilinear',                  'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !BILINEAR image:1920,1080,U008'),
    ('WarpAffine_U8_U8_Bilinear_constant',         'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('WarpPerspective_U8_U8_Nearest',              'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !NEAREST_NEIGHBOR image:1920,1080,U008'),
    ('WarpPerspective_U8_U8_Nearest_constant',     'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('WarpPerspective_U8_U8_Bilinear',             'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !BILINEAR image:1920,1080,U008'),
    ('WarpPerspective_U8_U8_Bilinear_constant',    'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    #('Remap_S16_nearest',                         'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008'),
    #('Remap_S16_nearest_constant',                'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    #('Remap_S16_bilinear',                        'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !BILINEAR image:1920,1080,U008'),
    #('Remap_S16_bilinear_constant',               'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
# vision kernels
    #('FastCorners_XY_U8_NoSupression',            'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,1 array:KEYPOINT,1000 scalar:SIZE,0'),
    #('FastCorners_XY_U8_Supression',              'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,0 array:KEYPOINT,1000 scalar:SIZE,0'),
    ('Lut_U8_U8',                                  'org.khronos.openvx.table_lookup image:1920,1080,U008 lut:UINT8,256 image:1920,1080,U008')
]

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--runvx_directory',    type=str, default='',
                    help='RunVX Executable directory - required')
parser.add_argument('--hardware_mode',      type=str, default='CPU',
                    help='OpenVX Vision Function target - optional (default:CPU [options:CPU/GPU])')
parser.add_argument('--backend_type',      type=str, default='HOST',
                    help='Backend type - optional (default:HOST [options:HOST/HIP/OCL])')
parser.add_argument('--profiling',      type=str, default='NO',
                    help='GPU profiling with rocprof - optional (default:NO [options:YES/NO])')
args = parser.parse_args()

runvxDir = args.runvx_directory
hardwareMode = args.hardware_mode
backendType = args.backend_type
profilingOption = args.profiling

if hardwareMode == "CPU":
    if backendType != "HOST" or profilingOption != "NO":
        print("For hardware_mode=CPU, the backend_type must be HOST and profiling must be NO")
        end()
if hardwareMode == "GPU":
    if backendType == "HOST":
        print("For hardware_mode=GPU, the backend_type must be either HIP or OCL")
        end()

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

# All cases with GPU profiling
if hardwareMode == "GPU" and profilingOption == "YES":
    os.system('rm -rvf rocprof_vision_tests_outputs')
    os.system('mkdir rocprof_vision_tests_outputs')
    totalCount = 0
    for i in range(len(openvxNodes[:])):
    # for i in range(len(openvxNodes[:]) - 20, len(openvxNodes[:]), 1):
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        os.system('mkdir rocprof_vision_tests_outputs/case_'+str(i+1))
        if backendType == "HIP":
            print('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:1000 -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
            os.system('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:1000 -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
        elif backendType == "OCL":
            print('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:1000 -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
            os.system('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:1000 -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
        print("\n")
        totalCount = i+1
    RESULTS_DIR = "rocprof_vision_tests_outputs"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"
    new_file = open(CONSOLIDATED_FILE,'w')
    if backendType == "HIP":
        new_file.write('"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')
    elif backendType == "OCL":
        new_file.write('"Kernel","Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')
    for case_num in range(1,totalCount+1,1):
        nodeName, nodeFormat = openvxNodes[case_num-1]
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(case_num)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)
        CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case_" + str(case_num) + ".stats.csv"
        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
        try:
            case_file = open(CASE_FILE_PATH,'r')
            if backendType == "HIP":
                for line in case_file:
                    print(line)
                    if line.startswith('"Hip'):
                        new_file.write(line)
            elif backendType == "OCL":
                for line in case_file:
                    print(line)
                    if line.startswith('"OpenVX_kernel'):
                        new_file.write('Ocl_'+nodeName+","+line)
            case_file.close()
        except:
            print("Unable to open case results")
            continue
    new_file.close()
    os.system('chown $USER:$USER '+RESULTS_DIR+'/consolidated_results.stats.csv')

# All cases without profiling
else:
    for i in range(len(openvxNodes[:])):
    # for i in range(len(openvxNodes[:]) - 30, len(openvxNodes[:]) - 28, 1):
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        print('rocprof --basenames on --timestamp on --stats  '+RunVXapp+' -frames:1000 -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
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