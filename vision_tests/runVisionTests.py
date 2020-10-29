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
    output = p.communicate()[0][0:-1]
    return output


def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


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

# OpenVX 1.0.1 Vision Functions 1080P
openvxNodes = [
    ('absdiff-1080p-u8', 'org.khronos.openvx.absdiff image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('accumulate-1080p-u8',
     'org.khronos.openvx.accumulate image:1920,1080,U008 image:1920,1080,S016'),
    ('accumulate_square-1080p-u8',
     'org.khronos.openvx.accumulate_square image:1920,1080,U008 scalar:UINT32,0 image:1920,1080,S016'),
    ('accumulate_weighted-1080p-u8',
     'org.khronos.openvx.accumulate_weighted image:1920,1080,U008 scalar:FLOAT32,0.3 image:1920,1080,U008'),
    ('add-1080p-u8', 'org.khronos.openvx.add image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('and-1080p-u8', 'org.khronos.openvx.and image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('box_3x3-1080p-u8',
     'org.khronos.openvx.box_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('canny_edge_detector-1080p-u8',
     'org.khronos.openvx.canny_edge_detector image:1920,1080,U008 threshold:RANGE,UINT8:INIT,80,100 scalar:INT32,3 !NORM_L1 image:1920,1080,U008'),
    ('channel_combine-1080p-u8',
     'org.khronos.openvx.channel_combine image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,RGB2'),
    ('dilate_3x3-1080p-u8',
     'org.khronos.openvx.dilate_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('erode_3x3-1080p-u8',
     'org.khronos.openvx.erode_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('fast_corners-1080p-u8', 'org.khronos.openvx.fast_corners image:1920,1080,U008 scalar:FLOAT32,80.0 scalar:BOOL,1 array:KEYPOINT,1000 scalar:UINT32,0'),
    ('gaussian_3x3-1080p-u8',
     'org.khronos.openvx.gaussian_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('gaussian_pyramid-1080p-u8',
     'org.khronos.openvx.gaussian_pyramid image:1920,1080,U008 pyramid:4,HALF,1920,1080,U008'),
    ('halfscale_gaussian-1080p-u8',
     'org.khronos.openvx.halfscale_gaussian image:1920,1080,U008 image:960,540,U008 scalar:INT32,5'),
    ('harris_corners-1080p-u8', 'org.khronos.openvx.harris_corners image:1920,1080,U008 scalar:FLOAT32,0.00001 scalar:FLOAT32,20.0 scalar:FLOAT32,0.10 scalar:INT32,3 scalar:INT32,5 array:KEYPOINT,1000 scalar:UINT32,0'),
    ('histogram-1080p-u8',
     'org.khronos.openvx.histogram image:1920,1080,U008 distribution:2,0,256'),
    ('integral_image-1080p-u8',
     'org.khronos.openvx.integral_image image:1920,1080,U008 image:1920,1080,U032'),
    ('magnitude-1080p-S16',
     'org.khronos.openvx.magnitude image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
    ('mean_stddev-1080p-u8',
     'org.khronos.openvx.mean_stddev image:1920,1080,U008 scalar:FLOAT32,0.0 scalar:FLOAT32,0.0'),
    ('median_3x3-1080p-u8',
     'org.khronos.openvx.median_3x3 image:1920,1080,U008 image:1920,1080,U008'),
    ('minmaxloc-1080p-u8',
     'org.khronos.openvx.minmaxloc image:1920,1080,U008 scalar:UINT8,0 scalar:UINT8,0 array:COORDINATES2D,1000 array:COORDINATES2D,1000 scalar:UINT32,0 scalar:UINT32,0'),
    ('multiply-1080p-u8', 'org.khronos.openvx.multiply image:1920,1080,U008 image:1920,1080,U008 scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:1920,1080,S016'),
    ('not-1080p-u8', 'org.khronos.openvx.not image:1920,1080,U008 image:1920,1080,U008'),
    ('optical_flow_pyr_lk-1080p-u8',
     'org.khronos.openvx.optical_flow_pyr_lk pyramid:4,HALF,1920,1080,U008 pyramid:4,HALF,1920,1080,U008 array:KEYPOINT,1000 array:KEYPOINT,1000 array:KEYPOINT,1000 scalar:ENUM,CRITERIA_BOTH scalar:FLOAT32,0.1 scalar:UINT32,4 scalar:BOOL,0 scalar:SIZE,5'),
    ('or-1080p-u8', 'org.khronos.openvx.or image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('phase-1080p-S16',
     'org.khronos.openvx.phase image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,U008'),
    ('remap-1080p-S16',
     'org.khronos.openvx.remap image:1920,1080,U008 remap:1920,1080,1920,1080 !NEAREST_NEIGHBOR image:1920,1080,U008 !BORDER_MODE_CONSTANT'),
    ('scale_image-1080p-u8',
     'org.khronos.openvx.scale_image image:1920,1080,U008 image:960,540,U008 !BILINEAR'),
    ('sobel_3x3-1080p-u8',
     'org.khronos.openvx.sobel_3x3 image:1920,1080,U008 image:1920,1080,S016 image:1920,1080,S016'),
    ('subtract-1080p-u8',
     'org.khronos.openvx.subtract image:1920,1080,U008 image:1920,1080,U008 !WRAP image:1920,1080,S016'),
    ('table_lookup-1080p-u8',
     'org.khronos.openvx.table_lookup image:1920,1080,U008 lut:UINT8,256 image:1920,1080,U008'),
    ('threshold-1080p-u8',
     'org.khronos.openvx.threshold image:1920,1080,U008 threshold:BINARY,UINT8:INIT,127 image:1920,1080,U008'),
    ('warp_affine-1080p-u8',
     'org.khronos.openvx.warp_affine image:1920,1080,U008 matrix:FLOAT32,2,3 !BILINEAR image:1920,1080,U008'),
    ('warp_perspective-1080p-u8',
     'org.khronos.openvx.warp_perspective image:1920,1080,U008 matrix:FLOAT32,3,3 !NEAREST_NEIGHBOR image:1920,1080,U008'),
    ('xor-1080p-u8', 'org.khronos.openvx.xor image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008')
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

#print("\nrunVisionTests - OpenVX Node Tests V-"+__version__+"\n")
#os.system('mkdir openvx_node_results')
# for i in range(len(openvxNodeTestConfig)):
    #nodeTestName, nodeTest = openvxNodeTestConfig[i]
    #print("Running OpenVX Node: "+nodeTestName)
    #os.system('./'+runvx_exe_dir+' -frames:1000 -affinity:'+hardwareMode+' -dump-profile node '+nodeTest+' | tee -a openvx_node_results/nodeTestOutput.log')
    # print("\n")

print("\nrunVisionTests - OpenVX Node Performance V-"+__version__+"\n")
outputDirectory = 'openvx_node_results'
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
else:
    shutil.rmtree(outputDirectory)
    os.makedirs(outputDirectory)
for i in range(len(openvxNodes)):
    nodeName, nodeFormat = openvxNodes[i]
    echo1 = 'Running OpenVX Node - '+nodeName
    os.system('echo '+echo1 +
              ' | tee -a openvx_node_results/nodePerformanceOutput.log')
    os.system('./'+runvx_exe_dir+' -frames:1000 -affinity:' +
              hardwareMode+' -dump-profile node '+nodeFormat+' | tee -a openvx_node_results/nodePerformanceOutput.log')
    print("\n")


orig_stdout = sys.stdout
sys.stdout = open('openvx_node_results/nodePerformance.md','a')
echo_1 = '| OpenVX Node | Frames Count | tmp (ms) | avg (ms) | min (ms) | max (ms) |'
print(echo_1)
echo_2 = '|-------------|--------------|----------|----------|----------|----------|'
print(echo_2)
sys.stdout = orig_stdout
print(echo_1)
print(echo_2)
runAwk_csv = r'''awk 'BEGIN { node = "xxx"; } /Running OpenVX Node - / { node = $5; } /CPU,GRAPH/ { printf("| %-30s | %3d | %8.3f | %8.3f | %8.3f | %8.3f |\n", node, $1, $2, $3, $4, $5) }' openvx_node_results/nodePerformanceOutput.log | tee -a openvx_node_results/nodePerformance.md'''
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
