# Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
import shutil
import sys
import platform

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2021, AMD MIVisionX - Vision Test Full Report"
__license__ = "MIT"
__version__ = "1.2.2"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiriti.NageshGowda@amd.com"
__status__ = "Shipping"

def shell(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output = p.communicate()[0][0:-1]
    return output

def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")

def strip_libtree_addresses(lib_tree):
    return lib_tree


# Vision Accuracy Tests
visionTestConfig = [
    '01_absDiff.gdf',
    '02_accumulate.gdf',
    '03_accumulateSquared.gdf',
    '04_accumulateWeighted.gdf',
    '05_add.gdf',
    '06_and.gdf',
    '07_box.gdf',
    '08_canny.gdf',
    '09_channelCombine.gdf',
    '10_channelExtract.gdf',
    '11_colorConvert.gdf',
    '12_convertDepth.gdf',
    '13_convolve.gdf',
    '14_dilate.gdf',
    '15_equalizeHistogram.gdf',
    '16_erode.gdf',
    '17_fastCorners.gdf',
    '18_gaussian.gdf',
    '19_harrisCorners.gdf',
    '20_halfScaleGaussian.gdf',
    '21_histogram.gdf',
    '22_integralImage.gdf',
    '23_magnitude.gdf',
    '24_meanStdDev.gdf',
    '25_median.gdf',
    '26_minMaxLoc.gdf',
    '27_multiply.gdf',
    '28_not.gdf',
    '29_opticalFlowLK.gdf',
    '30_or.gdf',
    '31_phase.gdf',
    '32_gaussianPyramid.gdf',
    '33_remap.gdf',
    '34_scaleImage.gdf',
    '35_sobel.gdf',
    '36_subtract.gdf',
    '37_tableLookup.gdf',
    '38_threshold.gdf',
    '39_warpAffine.gdf',
    '40_warpPerspective.gdf',
    '41_xor.gdf'
]

# OpenVX Vision Functions 1080P

openvxNodes = [
# Arithmetic kernels
    ('AbsDiff_U8_U8U8',                           'org.khronos.openvx.absdiff uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb image:1920,1080,U008'),
    ('AbsDiff_S16_S16S16_Sat',                    'org.khronos.openvx.absdiff uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb image:1920,1080,S016'),
    ('Add_U8_U8U8_Wrap',                          'org.khronos.openvx.add uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb !WRAP image:1920,1080,U008'),
    ('Add_U8_U8U8_Sat',                           'org.khronos.openvx.add uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb !SATURATE image:1920,1080,U008'),
    ('Add_S16_U8U8_Wrap',                         'org.khronos.openvx.add uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb !WRAP image:1920,1080,S016'),
    ('Add_S16_S16U8_Wrap',                        'org.khronos.openvx.add uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb !WRAP image:1920,1080,S016'),
    ('Add_S16_S16U8_Sat',                         'org.khronos.openvx.add uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb !SATURATE image:1920,1080,S016'),
    ('Add_S16_S16S16_Wrap',                       'org.khronos.openvx.add uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb !WRAP image:1920,1080,S016'),
    ('Add_S16_S16S16_Sat',                        'org.khronos.openvx.add uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb !SATURATE image:1920,1080,S016'),
    ('Sub_U8_U8U8_Wrap',                          'org.khronos.openvx.subtract uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb !WRAP image:1920,1080,U008'),
    ('Sub_U8_U8U8_Sat',                           'org.khronos.openvx.subtract uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb !SATURATE image:1920,1080,U008'),
    ('Sub_S16_U8U8_Wrap',                         'org.khronos.openvx.subtract uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb !WRAP image:1920,1080,S016'),
    ('Sub_S16_S16U8_Wrap',                        'org.khronos.openvx.subtract uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb !WRAP image:1920,1080,S016'),
    ('Sub_S16_S16U8_Sat',                         'org.khronos.openvx.subtract uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb !SATURATE image:1920,1080,S016'),
    ('Sub_S16_U8S16_Wrap',                        'org.khronos.openvx.subtract uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,S016,0xbb !WRAP image:1920,1080,S016'),
    ('Sub_S16_U8S16_Sat',                         'org.khronos.openvx.subtract uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,S016,0xbb !SATURATE image:1920,1080,S016'),
    ('Sub_S16_S16S16_Wrap',                       'org.khronos.openvx.subtract uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb !WRAP image:1920,1080,S016'),
    ('Sub_S16_S16S16_Sat',                        'org.khronos.openvx.subtract uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb !SATURATE image:1920,1080,S016'),
    ('Mul_U8_U8U8_Wrap_Trunc',                    'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
    ('Mul_U8_U8U8_Wrap_Round',                    'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
    ('Mul_U8_U8U8_Sat_Trunc',                     'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,U008'),
    ('Mul_U8_U8U8_Sat_Round',                     'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,U008'),
    ('Mul_S16_U8U8_Wrap_Trunc',                   'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_U8U8_Wrap_Round',                   'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_U8U8_Sat_Trunc',                    'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_U8U8_Sat_Round',                    'org.khronos.openvx.multiply uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16U8_Wrap_Trunc',                  'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16U8_Wrap_Round',                  'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16U8_Sat_Trunc',                   'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16U8_Sat_Round',                   'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16S16_Wrap_Trunc',                 'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16S16_Wrap_Round',                 'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Mul_S16_S16S16_Sat_Trunc',                  'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:1920,1080,S016'),
    ('Mul_S16_S16S16_Sat_Round',                  'org.khronos.openvx.multiply uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('Magnitude_S16_S16S16',                      'org.khronos.openvx.magnitude uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb image:1920,1080,S016'),
    ('Phase_U8_S16S16',                           'org.khronos.openvx.phase uniform-image:1920,1080,S016,0xaa uniform-image:1920,1080,S016,0xbb image:1920,1080,U008'),
    ('WeightedAverage_U8_U8U8',                   'org.khronos.openvx.weighted_average uniform-image:1920,1080,U008,0xaa scalar:FLOAT32,0.25 uniform-image:1920,1080,U008,0xbb image:1920,1080,U008'),
# logical kernels
    ('And_U8_U8U8',                               'org.khronos.openvx.and uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb image:1920,1080,U008'),
    ('Or_U8_U8U8',                                'org.khronos.openvx.or uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb image:1920,1080,U008'),
    ('Xor_U8_U8U8',                               'org.khronos.openvx.xor uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb image:1920,1080,U008'),
    ('Not_U8_U8',                                 'org.khronos.openvx.not uniform-image:1920,1080,U008,0xaa image:1920,1080,U008'),
# Color kernels
    ('Lut_U8_U8',                                 'org.khronos.openvx.table_lookup uniform-image:1920,1080,U008,0xaa lut:UINT8,256 image:1920,1080,U008'),
    ('ColorDepth_U8_S16_Wrap',                    'org.khronos.openvx.convertdepth uniform-image:1920,1080,S016,0xaa image:1920,1080,U008 !WRAP scalar:INT32,1'),
    ('ColorDepth_U8_S16_Sat',                     'org.khronos.openvx.convertdepth uniform-image:1920,1080,S016,0xaa image:1920,1080,U008 !SATURATE scalar:INT32,1'),
    ('ColorDepth_S16_U8',                         'org.khronos.openvx.convertdepth uniform-image:1920,1080,U008,0xaa image:1920,1080,S016 !WRAP scalar:INT32,1'),
    # ('ChannelExtract_U8_U16_Pos0',                'org.khronos.openvx.channel_extract uniform-image:1920,1080,YUYV !CHANNEL_Y image:1920,1080,U008'),
    # ('ChannelExtract_U8_U16_Pos1',                'org.khronos.openvx.channel_extract uniform-image:1920,1080,YUYV !CHANNEL_Y image:1920,1080,U008'),
    ('ChannelExtract_U8_U24_Pos0',                'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGB2,0xaabbcc !CHANNEL_R image:1920,1080,U008'),
    ('ChannelExtract_U8_U24_Pos1',                'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGB2,0xaabbcc !CHANNEL_G image:1920,1080,U008'),
    ('ChannelExtract_U8_U24_Pos2',                'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGB2,0xaabbcc !CHANNEL_B image:1920,1080,U008'),
    ('ChannelExtract_U8_U32_Pos0_UYVY',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,UYVY,0xaabbcc !CHANNEL_U image:960,1080,U008'),
    ('ChannelExtract_U8_U32_Pos1_YUYV',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,YUYV,0xaabbcc !CHANNEL_U image:960,1080,U008'),
    ('ChannelExtract_U8_U32_Pos2_UYVY',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,UYVY,0xaabbcc !CHANNEL_V image:960,1080,U008'),
    ('ChannelExtract_U8_U32_Pos3_YUYV',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,YUYV,0xaabbcc !CHANNEL_V image:960,1080,U008'),
    ('ChannelExtract_U8_U32_Pos0_RGBX',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGBA,0xaabbccdd !CHANNEL_R image:1920,1080,U008'),
    ('ChannelExtract_U8_U32_Pos1_RGBX',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGBA,0xaabbccdd !CHANNEL_G image:1920,1080,U008'),
    ('ChannelExtract_U8_U32_Pos2_RGBX',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGBA,0xaabbccdd !CHANNEL_B image:1920,1080,U008'),
    ('ChannelExtract_U8_U32_Pos3_RGBX',           'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGBA,0xaabbccdd !CHANNEL_A image:1920,1080,U008'),
    ('ChannelExtract_U8U8U8_U24',                 'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGB2,0xaabbcc !CHANNEL_R image:1920,1080,U008'),
    ('ChannelExtract_U8U8U8_U32',                 'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGBA,0xaabbccdd !CHANNEL_R image:1920,1080,U008'),
    ('ChannelExtract_U8U8U8U8_U32',               'org.khronos.openvx.channel_extract uniform-image:1920,1080,RGBA,0xaabbccdd !CHANNEL_R image:1920,1080,U008'),
    # ('ChannelCombine_U16_U8U8',                   'org.khronos.openvx.channel_combine uniform-image:1920,1080,U008 uniform-image:960,540,U008 uniform-image:960,540,U008 image:1920,1080,NV12'),
    # ('ChannelCombine_U32_U8U8U8_UYVY',            'org.khronos.openvx.channel_combine uniform-image:1920,1080,U008,0xaa uniform-image:960,1080,U008,0xbb uniform-image:960,1080,U008,0xcc null image:1920,1080,UYVY'),
    # ('ChannelCombine_U32_U8U8U8_YUYV',            'org.khronos.openvx.channel_combine uniform-image:1920,1080,U008,0xaa uniform-image:960,1080,U008,0xbb uniform-image:960,1080,U008,0xcc null image:1920,1080,YUYV'),
    # ('ChannelCombine_U24_U8U8U8_RGB',             'org.khronos.openvx.channel_combine uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb uniform-image:1920,1080,U008,0xcc null image:1920,1080,RGB2'),
    ('ChannelCombine_U32_U8U8U8U8_RGBX',          'org.khronos.openvx.channel_combine uniform-image:1920,1080,U008,0xaa uniform-image:1920,1080,U008,0xbb uniform-image:1920,1080,U008,0xcc uniform-image:1920,1080,U008,0xdd image:1920,1080,RGBA'),
    ('ColorConvert_RGB_RGBX',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,RGBA,0xaabbccdd image:1920,1080,RGB2'),
    ('ColorConvert_RGB_UYVY',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,UYVY,0xaabbcc image:1920,1080,RGB2'),
    ('ColorConvert_RGB_YUYV',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,YUYV,0xaabbcc image:1920,1080,RGB2'),
    ('ColorConvert_RGB_IYUV',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,IYUV,0xaabbcc image:1920,1080,RGB2'),
    ('ColorConvert_RGB_NV12',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,NV12,0xaabbcc image:1920,1080,RGB2'),
    ('ColorConvert_RGB_NV21',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,NV21,0xaabbcc image:1920,1080,RGB2'),
    ('ColorConvert_RGBX_RGB',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,RGB2,0xaabbcc image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_UYVY',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,UYVY,0xaabbcc image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_YUYV',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,YUYV,0xaabbcc image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_IYUV',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,IYUV,0xaabbcc image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_NV12',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,NV12,0xaabbcc image:1920,1080,RGBA'),
    ('ColorConvert_RGBX_NV21',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,NV21,0xaabbcc image:1920,1080,RGBA'),
    ('ColorConvert_IYUV_RGB',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,RGB2,0xaabbcc image:1920,1080,IYUV'),
    ('ColorConvert_IYUV_RGBX',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,RGBA,0xaabbccdd image:1920,1080,IYUV'),
    ('FormatConvert_IYUV_UYVY',                   'org.khronos.openvx.color_convert uniform-image:1920,1080,UYVY,0xaabbcc image:1920,1080,IYUV'),
    ('FormatConvert_IYUV_YUYV',                   'org.khronos.openvx.color_convert uniform-image:1920,1080,YUYV,0xaabbcc image:1920,1080,IYUV'),
    ('ColorConvert_NV12_RGB',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,RGB2,0xaabbcc image:1920,1080,NV12'),
    ('ColorConvert_NV12_RGBX',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,RGBA,0xaabbccdd image:1920,1080,NV12'),
    ('FormatConvert_NV12_UYVY',                   'org.khronos.openvx.color_convert uniform-image:1920,1080,UYVY,0xaabbcc image:1920,1080,NV12'),
    ('FormatConvert_NV12_YUYV',                   'org.khronos.openvx.color_convert uniform-image:1920,1080,YUYV,0xaabbcc image:1920,1080,NV12'),
    ('ColorConvert_YUV4_RGB',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,RGB2,0xaabbcc image:1920,1080,YUV4'),
    ('ColorConvert_YUV4_RGBX',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,RGBA,0xaabbccdd image:1920,1080,YUV4'),
    # ('FormatConvert_IUV_UV12',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,NV12,0xaabbcc image:1920,1080,IYUV'),
    # ('FormatConvert_UV12_IUV',                    'org.khronos.openvx.color_convert uniform-image:1920,1080,IYUV,0xaabbcc image:1920,1080,NV12'),
    # ('FormatConvert_UV_UV12',                     'org.khronos.openvx.color_convert uniform-image:1920,1080,NV12,0xaabbcc image:1920,1080,YUV4'),
    # ('ScaleUp2x2_U8_U8',                          'org.khronos.openvx.color_convert uniform-image:1920,1080,IYUV,0xaabbcc image:1920,1080,YUV4'),
# filter kernels
    ('Box_U8_U8_3x3',                             'org.khronos.openvx.box_3x3 uniform-image:1920,1080,U008,0xaa image:1920,1080,U008'),
    ('Dilate_U8_U8_3x3',                          'org.khronos.openvx.dilate_3x3 uniform-image:1920,1080,U008,0xaa image:1920,1080,U008'),
    ('Erode_U8_U8_3x3',                           'org.khronos.openvx.erode_3x3 uniform-image:1920,1080,U008,0xaa image:1920,1080,U008'),
    ('Median_U8_U8_3x3',                          'org.khronos.openvx.median_3x3 uniform-image:1920,1080,U008,0xaa image:1920,1080,U008'),
    ('Gaussian_U8_U8_3x3',                        'org.khronos.openvx.gaussian_3x3 uniform-image:1920,1080,U008,0xaa image:1920,1080,U008'),
    ('ScaleGaussianHalf_U8_U8_3x3',               'org.khronos.openvx.halfscale_gaussian uniform-image:1920,1080,U008,0xaa image:960,540,U008 scalar:INT32,3'),
    ('ScaleGaussianHalf_U8_U8_5x5',               'org.khronos.openvx.halfscale_gaussian uniform-image:1920,1080,U008,0xaa image:960,540,U008 scalar:INT32,5'),
    ('Convolve_U8_U8_3x3',                        'org.khronos.openvx.custom_convolution uniform-image:1920,1080,U008,0xaa "convolution:3,3:INIT,{-1;-1;-1;-1;16;-1;-1;-1;-1}" image:1920,1080,U008'),
    ('Convolve_S16_U8_3x3',                       'org.khronos.openvx.custom_convolution uniform-image:1920,1080,U008,0xaa "convolution:3,3:INIT,{-1;-1;-1;-1;16;-1;-1;-1;-1}" image:1920,1080,S016'),
    ('Sobel_S16S16_U8_3x3_GXY',                   'org.khronos.openvx.sobel_3x3 uniform-image:1920,1080,U008,0xaa image:1920,1080,S016 image:1920,1080,S016'),
    ('Sobel_S16_U8_3x3_GX',                       'org.khronos.openvx.sobel_3x3 uniform-image:1920,1080,U008,0xaa image:1920,1080,S016'),
    # ('Sobel_S16_U8_3x3_GY',                       'org.khronos.openvx.sobel_3x3 uniform-image:1920,1080,U008,0xaa null image:1920,1080,S016'),
# Statistical kernels
    ('Threshold_U8_U8_Binary',                    'org.khronos.openvx.threshold uniform-image:1920,1080,U008,0xaa threshold:BINARY,U008,U008:INIT,127 image:1920,1080,U008'),
    ('Threshold_U8_U8_Range',                     'org.khronos.openvx.threshold uniform-image:1920,1080,U008,0xaa threshold:RANGE,U008,U008:INIT,100,200 image:1920,1080,U008'),
    ('Threshold_U8_S16_Binary',                   'org.khronos.openvx.threshold uniform-image:1920,1080,S016,0xaa threshold:BINARY,S016,U008:INIT,127 image:1920,1080,U008'),
    ('Threshold_U8_S16_Range',                    'org.khronos.openvx.threshold uniform-image:1920,1080,S016,0xaa threshold:RANGE,S016,U008:INIT,100,200 image:1920,1080,U008'),
# geometric kernels
    ('ScaleImage_U8_U8_Nearest',                  'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !NEAREST_NEIGHBOR'),
    ('ScaleImage_U8_U8_Bilinear',                 'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR'),
    ('ScaleImage_U8_U8_Bilinear_Replicate',       'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR attr:BORDER_MODE:REPLICATE'),
    ('ScaleImage_U8_U8_Bilinear_Constant',        'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR attr:BORDER_MODE:CONSTANT,0'),
    ('ScaleImage_U8_U8_Area',                     'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !AREA'),
    ('WarpAffine_U8_U8_Nearest',                  'org.khronos.openvx.warp_affine image:1920,1080,U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !NEAREST_NEIGHBOR image:1920,1080,U008'),
    ('WarpAffine_U8_U8_Nearest_Constant',         'org.khronos.openvx.warp_affine image:1920,1080,U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    ('WarpAffine_U8_U8_Bilinear',                 'org.khronos.openvx.warp_affine image:1920,1080,U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !BILINEAR image:1920,1080,U008'),
    ('WarpAffine_U8_U8_Bilinear_Constant',        'org.khronos.openvx.warp_affine image:1920,1080,U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    # ('WarpPerspective_U8_U8_Nearest',             'org.khronos.openvx.warp_perspective image:1920,1080,U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !NEAREST_NEIGHBOR image:1920,1080,U008'),
    # ('WarpPerspective_U8_U8_Nearest_Constant',    'org.khronos.openvx.warp_perspective image:1920,1080,U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
    # ('WarpPerspective_U8_U8_Bilinear',            'org.khronos.openvx.warp_perspective image:1920,1080,U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !BILINEAR image:1920,1080,U008'),
    # ('WarpPerspective_U8_U8_Bilinear_Constant',   'org.khronos.openvx.warp_perspective image:1920,1080,U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
# To be updated...
#     ('Remap_S16_nearest',                         'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008'),
#     ('Remap_S16_nearest_constant',                'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
#     ('Remap_S16_bilinear',                        'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !BILINEAR image:1920,1080,U008'),
#     ('Remap_S16_bilinear_constant',               'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !BILINEAR image:1920,1080,U008 attr:BORDER_MODE:CONSTANT,0'),
# # vision kernels
#     ('FastCorners_XY_U8_NoSupression',            'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,1 array:KEYPOINT,1000 scalar:SIZE,0'),
#     ('FastCorners_XY_U8_Supression',              'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,0 array:KEYPOINT,1000 scalar:SIZE,0'),
]

#  Popular Video Sizes
# { 2160p: 3840x2160, 1440p: 2560x1440, 1080p: 1920x1080, 720p: 1280x720, 480p: 854x480, 360p: 640x360, 240p: 426x240 }
openvxNodeTestConfig = [
    # absdiff U8 - U8
    ('absdiff-240p-u8', 'org.khronos.openvx.absdiff image:426,240,U008 image:426,240,U008 image:426,240,U008'),
    ('absdiff-360p-u8', 'org.khronos.openvx.absdiff image:640,360,U008 image:640,360,U008 image:640,360,U008'),
    ('absdiff-480p-u8', 'org.khronos.openvx.absdiff image:854,480,U008 image:854,480,U008 image:854,480,U008'),
    ('absdiff-720p-u8', 'org.khronos.openvx.absdiff image:1280,720,U008 image:1280,720,U008 image:1280,720,U008'),
    ('absdiff-1080p-u8', 'org.khronos.openvx.absdiff image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('absdiff-1440p-u8', 'org.khronos.openvx.absdiff image:2560,1440,U008 image:2560,1440,U008 image:2560,1440,U008'),
    ('absdiff-2160p-u8', 'org.khronos.openvx.absdiff image:3840,2160,U008 image:3840,2160,U008 image:3840,2160,U008'),
    # absdiff S16 - S16
    ('absdiff-240p-s16', 'org.khronos.openvx.absdiff image:426,240,S016 image:426,240,S016 image:426,240,S016'),
    ('absdiff-360p-s16', 'org.khronos.openvx.absdiff image:640,360,S016 image:640,360,S016 image:640,360,S016'),
    ('absdiff-480p-s16', 'org.khronos.openvx.absdiff image:854,480,S016 image:854,480,S016 image:854,480,S016'),
    ('absdiff-720p-s16', 'org.khronos.openvx.absdiff image:1280,720,S016 image:1280,720,S016 image:1280,720,S016'),
    ('absdiff-1080p-s16', 'org.khronos.openvx.absdiff image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
    ('absdiff-1440p-s16', 'org.khronos.openvx.absdiff image:2560,1440,S016 image:2560,1440,S016 image:2560,1440,S016'),
    ('absdiff-2160p-s16', 'org.khronos.openvx.absdiff image:3840,2160,S016 image:3840,2160,S016 image:3840,2160,S016')
]

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--runvx_directory',    type=str, default='',
                    help='RunVX Executable Directory - required')
parser.add_argument('--hardware_mode',      type=str, default='CPU',
                    help='OpenVX Vision Function Target - optional (default:CPU [options:CPU/GPU])')
parser.add_argument('--list_tests',         type=str, default='no',
                    help='List Vision Performance Tests - optional (default:no [options:no/yes])')
parser.add_argument('--test_filter',        type=int, default=0,
                    help='Vision Performance Test Filter - optional (default:0 [range:1 - N])')
parser.add_argument('--num_frames',         type=int, default=1000,
                    help='Run Test for X number of frames - optional (default:1000 [range:1 - N])')
parser.add_argument('--functionality',      type=str, default='yes',
                    help='Vision Functionality Tests Enabled - optional (default:yes [options:no/yes])')
parser.add_argument('--backend_type',       type=str, default='HOST',
                    help='Backend type - optional (default:HOST [options:HOST/HIP/OCL])')
parser.add_argument('--profiling',          type=str, default='no',
                    help='GPU profiling with rocprof - optional (default:no [options:yes/no])')
args = parser.parse_args()

runvxDir = args.runvx_directory
hardwareMode = args.hardware_mode
listTest = args.list_tests
testFilter = args.test_filter
numFrames = args.num_frames
functionalityTests = args.functionality
backendType = args.backend_type
profilingOption = args.profiling

# check arguments
if hardwareMode not in ('CPU', 'GPU'):
    print("ERROR: OpenVX Hardware supported - CPU or GPU]")
    exit()
if listTest not in ('no', 'yes'):
    print("ERROR: List Vision Performance Tests options supported - [no or yes]")
    exit()
if functionalityTests not in ('no', 'yes'):
    print("ERROR: Vision functionality Tests option supported - [no or yes]")
    exit()
if backendType not in ('HOST', 'HIP', 'OCL'):
    print("ERROR: OpenVX Backends supported - HOST or HIP or OCL]")
    exit()
if profilingOption not in ('no', 'yes'):
    print("ERROR: Profiling options supported - [no or yes]")
    exit()

if not 0 <= testFilter <= len(openvxNodes):
    print("\nERROR: Vision Performance Filter not in range - [1 - %d]\n" % (len(openvxNodes)))
    exit()
if not 1 <= numFrames <= 10000:
    print("\nERROR: Vision Test Number of Frames not in range - [1 - 10000]\n")
    exit()

if hardwareMode == "CPU":
    if backendType != "HOST" or profilingOption != "no":
        print("For hardware_mode=CPU, the backend_type must be 'HOST' and profiling must be 'no'")
        exit()
if hardwareMode == "GPU":
    if backendType == "HOST":
        print("For hardware_mode=GPU, the backend_type must be either 'HIP' or 'OCL'")
        exit()

backendTypeValue = 0
if backendType == "OCL":
    backendTypeValue = 0
elif backendType == "HIP":
    backendTypeValue = 1

# List Vision Functionality tests
if listTest == 'yes':
    print(" %-5s - %-30s\n" % ('Test ID', 'Test Name'))
    for i in range(len(openvxNodes)):
        nodeName, nodeFormat = openvxNodes[i]
        print("   %-5d - %-30s\n" % ((i+1), nodeName))
    exit()
if runvxDir == '':
    print("\nERROR: RunVX Executable Directory Required\n")
    print("USAGE: python runVisionTests.py --help\n")
    exit()

print("\nMIVisionX runVisionTests V-"+__version__+"\n")

# RunVX Application
runVX_exe = runvxDir+'/runvx'
RunVXapp = os.path.abspath(runVX_exe)
scriptPath = os.path.dirname(os.path.realpath(__file__))
if(os.path.isfile(RunVXapp)):
    print("STATUS: RunVX path - "+RunVXapp)
else:
    print("\nERROR: RunVX Executable Not Found\n")
    exit()

# Get cwd
cwd = os.getcwd()

# OpenVX Vision Functionality Tests

if testFilter == 0 and functionalityTests == 'yes':
    # create directory to store vision accurarcy test results
    os.system('(cd '+scriptPath+'/gdfs; mkdir -p openvx_test_results)')
    print("\nrunVisionTests - OpenVX Vision Functionality Tests\n")
    for i in range(len(visionTestConfig)):
        testFileName = visionTestConfig[i]
        print("Running Test Script: "+testFileName)
        os.system(RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                  hardwareMode+' -dump-profile file '+scriptPath+'/gdfs/'+testFileName+' | tee -a '+scriptPath+'/gdfs/openvx_test_results/VisionOutput.log')
        print("\n")
    print("\nSTATUS: Vision Accuracy Results - " +
          scriptPath+"/gdfs/openvx_test_results\n")

# OpenVX Performance Tests
print("\nrunVisionTests - OpenVX Node Performance\n")
outputDirectory = 'openvx_node_results'
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
else:
    shutil.rmtree(outputDirectory)
    os.makedirs(outputDirectory)

# Option A - All cases / single case with GPU profiling

nodeList = []
case_num_list = []

def multiCaseProfilerOCL(nodeList, case_num_list):

    for i in nodeList:
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        os.system('mkdir '+cwd+'/rocprof_vision_tests_outputs/case_'+str(i+1))
        print('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
            hardwareMode+' -dump-profile node '+nodeFormat)
        os.system('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
            hardwareMode+' -dump-profile node '+nodeFormat)
        print("\n")

    RESULTS_DIR = "rocprof_vision_tests_outputs"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"
    new_file = open(CONSOLIDATED_FILE,'w')
    new_file.write('"OCL Kernel Name","Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

    for case_num in case_num_list:
        nodeName, nodeFormat = openvxNodes[case_num-1]
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(case_num)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)
        CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case_" + str(case_num) + ".stats.csv"
        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
        try:
            case_file = open(CASE_FILE_PATH,'r')
            for line in case_file:
                print(line)
                if line.startswith('"OpenVX_kernel'):
                    new_file.write('Ocl_'+nodeName+","+line)
            case_file.close()
        except IOError:
            print("Unable to open case results")
            continue

    new_file.close()
    os.system('chown $USER:$USER '+RESULTS_DIR+'/consolidated_results.stats.csv')

    try:
        import pandas as pd
        pd.options.display.max_rows = None
        df = pd.read_csv(CONSOLIDATED_FILE)
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Name', 'Percentage'], axis=1)
        dfPrint["OCL Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Ocl_")
        print(dfPrint)
    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in " + CONSOLIDATED_FILE)
    except IOError:
            print("Unable to open results in " + CONSOLIDATED_FILE)

def multiCaseProfilerHIP(nodeList, case_num_list):

    for i in nodeList:
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        os.system('mkdir '+cwd+'/rocprof_vision_tests_outputs/case_'+str(i+1))
        print('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
            hardwareMode+' -dump-profile node '+nodeFormat)
        os.system('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
            hardwareMode+' -dump-profile node '+nodeFormat)
        print("\n")

    RESULTS_DIR = "rocprof_vision_tests_outputs"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"
    new_file = open(CONSOLIDATED_FILE,'w')
    new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

    for case_num in case_num_list:
        nodeName, nodeFormat = openvxNodes[case_num-1]
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(case_num)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)
        CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case_" + str(case_num) + ".stats.csv"
        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
        try:
            case_file = open(CASE_FILE_PATH,'r')
            for line in case_file:
                print(line)
                if line.startswith('"Hip'):
                    new_file.write(line)
            case_file.close()
        except IOError:
            print("Unable to open case results")
            continue

    new_file.close()
    os.system('chown $USER:$USER '+RESULTS_DIR+'/consolidated_results.stats.csv')

    try:
        import pandas as pd
        pd.options.display.max_rows = None
        df = pd.read_csv(CONSOLIDATED_FILE)
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Percentage'], axis=1)
        dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
        print(dfPrint)
    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in " + CONSOLIDATED_FILE)
    except IOError:
            print("Unable to open results in " + CONSOLIDATED_FILE)

if testFilter == 0:
    totalCount = len(openvxNodes[:])
    nodeList = range(totalCount)
    case_num_list = range(1, totalCount + 1, 1)
else:
    totalCount = testFilter
    nodeList = range((totalCount - 1), totalCount, 1)
    case_num_list = range(totalCount, totalCount + 1, 1)

if profilingOption == "yes":

    os.system('rm -rvf '+cwd+'/rocprof_vision_tests_outputs')
    os.system('mkdir '+cwd+'/rocprof_vision_tests_outputs')

    if backendType == "OCL":
        multiCaseProfilerOCL(nodeList = nodeList, case_num_list = case_num_list)
    else:
        multiCaseProfilerHIP(nodeList = nodeList, case_num_list = case_num_list)

# Option B - All cases / single case without GPU profiling
else:
    for i in nodeList:
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        print(RunVXapp+' -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile node '+nodeFormat)
        os.system(RunVXapp+' -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile node '+nodeFormat+' | tee -a openvx_node_results/nodePerformanceOutput.log')
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
    platform_name = platform.platform()
    platform_name_fq = shell('hostname --all-fqdns')
    platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

    file_dtstr = datetime.now().strftime("%Y%m%d")
    reportFilename = 'vision_report_%s_%s_%s.md' % (
        platform_name, file_dtstr, hardwareMode)
    report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    sys_info = shell('inxi -c0 -S')

    cpu_info = shell('inxi -c0 -C')
    # cpu_info = cpu_info.split('\n')[0]  # strip out clock speeds

    gpu_info = shell('inxi -c0 -G')
    # gpu_info = gpu_info.split('\n')[0]  # strip out X info

    memory_info = shell('inxi -c 0 -m')
    board_info = shell('inxi -c0 -M')

    lib_tree = shell('ldd '+RunVXapp)
    lib_tree = strip_libtree_addresses(lib_tree)

    # Write Report
    with open(reportFilename, 'w') as f:
        f.write("MIVisionX - OpenVX Function Report\n")
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
        f.write("\n")
        f.write("### Hardware: %s\n" % hardwareMode)
        f.write("\n")
        with open('openvx_node_results/nodePerformance.md') as benchmarkFile:
            for line in benchmarkFile:
                f.write("%s" % line)
        f.write("\n")
        f.write("\n")
        f.write("Dynamic Libraries Report\n")
        f.write("-----------------\n")
        f.write("\n")
        write_formatted(lib_tree, f)
        f.write("\n")

        f.write("\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2020 -- runVisionTests.py V-"+__version__+"**\n")
        f.write("\n")

        # report file
        reportFileDir = os.path.abspath(reportFilename)
        print("\nSTATUS: Output Report File - "+reportFileDir)

print("\nrunVisionTests.py completed - V:"+__version__+"\n")