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
    os.system('(cd gdfs; ./../'+runvx_exe_dir+' -affinity:' +
              hardwareMode+' -frames:100 -dump-profile file '+testFileName+')')
    print("\n")

print("\nrunVisionTests.py completed - V:"+__version__+"\n")
