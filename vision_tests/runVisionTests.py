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

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018-2020, AMD MIVision Generate Full Report"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiriti.NageshGowda@amd.com"
__status__ = "Shipping"

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
runvx_exe_dir = os.path.expanduser(runVX_exe)

print("\nrunVisionTests - OpenVX Vision Tests V-"+__version__+"\n")
os.system('(cd gdfs; mkdir openvx_test_results)')
for i in range(len(visionTestConfig)):
    testFileName = visionTestConfig[i]
    print("Running Test Script: "+testFileName)
    os.system('(cd gdfs; ./../'+runvx_exe_dir+' -frames:100 -affinity:' +
              hardwareMode+' -dump-profile file '+testFileName+' | tee -a openvx_test_results/VisionOutput.log)')
    print("\n")

print("\nrunVisionTests - OpenVX Node Tests V-"+__version__+"\n")
os.system('mkdir openvx_node_results')
for i in range(len(openvxNodeTestConfig)):
    nodeTestName, nodeTest = openvxNodeTestConfig[i]
    print("Running OpenVX Node: "+nodeTestName)
    os.system('./'+runvx_exe_dir+' -frames:1000 -affinity:' +
              hardwareMode+' -dump-profile node '+nodeTest+' | tee -a openvx_node_results/nodeTestOutput.log')
    print("\n")

print("\nrunVisionTests.py completed - V:"+__version__+"\n")
