# Copyright (c) 2020 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from subprocess import Popen, PIPE
import argparse
import os
import traceback

__copyright__ = "Copyright 2018 - 2025, AMD MIVisionX - OpenVX GDF Test"
__license__ = "MIT"
__version__ = "1.3.0"
__email__ = "mivisionx.support@amd.com"
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

# error check calls
def ERROR_CHECK(waitval):
    if(waitval != 0): # return code and signal flags
        print('ERROR_CHECK failed with status:'+str(waitval))
        traceback.print_stack()
        status = ((waitval >> 8) | waitval) & 255 # combine exit code and wait flags into single non-zero byte
        exit(status)

def get_gdf_names(directory):
    file_names = []
    try:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                file_names.append(filename)
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
    return file_names

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--runvx_directory',    type=str, default='/opt/rocm/bin',
                    help='RunVX Executable Directory - optional (default:/opt/rocm/bin)')
parser.add_argument('--list_tests',         type=str, default='no',
                    help='List Vision Performance Tests - optional (default:no [options:no/yes])')
parser.add_argument('--num_frames',         type=int, default=100,
                    help='Run Test for X number of frames - optional (default:100 [range:1 - N])')
parser.add_argument('--backend_type',       type=str, default='CPU',
                    help='Backend type - optional (default:CPU [options:CPU/HIP/OCL])')
parser.add_argument('--hidden',             type=str, default='no',
                    help='Hidden Vision Tests - optional (default:no [options:no/yes])')


args = parser.parse_args()

runvxDir = args.runvx_directory
listTest = args.list_tests.upper()
numFrames = args.num_frames
backendType = args.backend_type.upper()
hiddenTests = args.hidden.upper()

# check arguments
if listTest not in ('NO', 'YES'):
    print(
        "ERROR: List Vision Performance Tests options supported - [NO or YES]")
    exit()
if backendType not in ('CPU', 'HIP', 'OCL', 'GPU'):
    print("ERROR: OpenVX Backends supported - CPU or HIP or OCL or GPU]")
    exit()
if not 1 <= numFrames <= 10000:
    print("\nERROR: Vision Test Number of Frames not in range - [1 - 10000]\n")
    exit()
if hiddenTests not in ('NO', 'YES'):
    print(
        "ERROR: Hidden Vision Tests - [NO or YES]")
    exit()
    
hardwareMode = 'GPU'
if backendType == 'CPU':
    hardwareMode = 'CPU'

scriptPath = os.path.dirname(os.path.realpath(__file__))
# arithmetic nodes
arithmeticDirectoryPath = scriptPath+'/arithmetic'
arithmeticNodes = get_gdf_names(arithmeticDirectoryPath)
arithmeticNodes.sort()
# color nodes
colorDirectoryPath = scriptPath+'/color'
colorNodes = get_gdf_names(colorDirectoryPath)
colorNodes.sort()
# filter nodes
filterDirectoryPath = scriptPath+'/filter'
filterNodes = get_gdf_names(filterDirectoryPath)
filterNodes.sort()
# geometric nodes
geometricDirectoryPath = scriptPath+'/geometric'
geometricNodes = get_gdf_names(geometricDirectoryPath)
geometricNodes.sort()
# logical nodes
logicalDirectoryPath = scriptPath+'/logical'
logicalNodes = get_gdf_names(logicalDirectoryPath)
logicalNodes.sort()
# statistical nodes
statisticalDirectoryPath = scriptPath+'/statistical'
statisticalNodes = get_gdf_names(statisticalDirectoryPath)
statisticalNodes.sort()
# vision nodes
visionDirectoryPath = scriptPath+'/vision'
visionNodes = get_gdf_names(visionDirectoryPath)
visionNodes.sort()
# Vision Profile Nodes
visionProfileDirectoryPath = scriptPath+'/vision_profile'
visionProfileNodes = get_gdf_names(visionProfileDirectoryPath)
visionProfileNodes.sort()
# CPU Nodes
cpuDirectoryPath = scriptPath+'/cpu'
cpuNodes = get_gdf_names(cpuDirectoryPath)
cpuNodes.sort()
# CPU Nodes
hiddenDirectoryPath = scriptPath+'/cpu/hidden'
hiddenNodes = get_gdf_names(hiddenDirectoryPath)
hiddenNodes.sort()

# List tests
if listTest == 'YES':
    if arithmeticNodes:
        print("\nArithmetic Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(arithmeticNodes)):
            print("   %-5d - %-30s" % ((i+1), arithmeticNodes[i]))
    if colorNodes:
        print("\nColor Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(colorNodes)):
            print("   %-5d - %-30s" % ((i+1), colorNodes[i]))
    if filterNodes:
        print("\nFilter Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(filterNodes)):
            print("   %-5d - %-30s" % ((i+1), filterNodes[i]))
    if geometricNodes:
        print("\nGeometric Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(geometricNodes)):
            print("   %-5d - %-30s" % ((i+1), geometricNodes[i]))
    if logicalNodes:
        print("\nLogical Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(logicalNodes)):
            print("   %-5d - %-30s" % ((i+1), logicalNodes[i]))
    if statisticalNodes:
        print("\nStatistical Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(statisticalNodes)):
            print("   %-5d - %-30s" % ((i+1), statisticalNodes[i]))
    if visionNodes:
        print("\nVision Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(visionNodes)):
            print("   %-5d - %-30s" % ((i+1), visionNodes[i]))
    if visionProfileNodes:
        print("\nVision Profile Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(visionProfileNodes)):
            print("   %-5d - %-30s" % ((i+1), visionProfileNodes[i]))
    if cpuNodes:
        print("\nCPU Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(cpuNodes)):
            print("   %-5d - %-30s" % ((i+1), cpuNodes[i]))
    if hiddenTests == 'YES' and hiddenNodes:
        print("\nHidden CPU Node Tests\n")
        print(" %-5s - %-30s" % ('Test ID', 'GDF Name'))
        for i in range(len(hiddenNodes)):
            print("   %-5d - %-30s" % ((i+1), hiddenNodes[i]))
    exit()

if runvxDir == '':
    print("\nERROR: RunVX Executable Directory Required\n")
    print("USAGE: python runVisionTests.py --help\n")
    exit()

print("\nMIVisionX runOpenVX V-"+__version__+"\n")

# RunVX Application
runVX_exe = runvxDir+'/runvx'
RunVXapp = os.path.abspath(runVX_exe)
if(os.path.isfile(RunVXapp)):
    print("STATUS: RunVX path - "+RunVXapp)
else:
    print("\nERROR: RunVX Executable Not Found\n")
    exit()

# OpenVX Vision Node Tests
print("\nrunOpenVX - OpenVX Vision GDF Tests\n")

for i in range(len(arithmeticNodes)):
    testFileName = arithmeticNodes[i]
    print("Arithmetic: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/arithmetic/'+testFileName))
    print("\n")
for i in range(len(colorNodes)):
    testFileName = colorNodes[i]
    print("Color:  Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/color/'+testFileName))
    print("\n")
for i in range(len(filterNodes)):
    testFileName = filterNodes[i]
    print("Filter: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/filter/'+testFileName))
    print("\n")
for i in range(len(geometricNodes)):
    testFileName = geometricNodes[i]
    print("Geometric: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/geometric/'+testFileName))
    print("\n")
for i in range(len(logicalNodes)):
    testFileName = logicalNodes[i]
    print("Logical: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/logical/'+testFileName))
    print("\n")
for i in range(len(statisticalNodes)):
    testFileName = statisticalNodes[i]
    print("Statistical: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/statistical/'+testFileName))
    print("\n")
for i in range(len(visionNodes)):
    testFileName = visionNodes[i]
    print("Vision: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/vision/'+testFileName))
    print("\n")
for i in range(len(visionProfileNodes)):
    testFileName = visionProfileNodes[i]
    print("Vision Apps: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/vision_profile/'+testFileName))
    print("\n")
for i in range(len(cpuNodes)):
    testFileName = cpuNodes[i]
    print("CPU Only: Running GDF - "+str(i+1)+':'+testFileName)
    ERROR_CHECK(os.system('echo '+testFileName))
    ERROR_CHECK(os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/cpu/'+testFileName))
    print("\n")
if hiddenTests == 'YES':
    for i in range(len(hiddenNodes)):
        testFileName = hiddenNodes[i]
        print("Failures: Running GDF - "+str(i+1)+':'+testFileName)
        ERROR_CHECK(os.system('echo '+testFileName))
        # Tests are not error checked -- Expected failures
        (os.system(RunVXapp+' -dump-gdf -frames:'+str(numFrames)+' -affinity:'+hardwareMode+' -dump-profile file '+scriptPath+'/cpu/hidden/'+testFileName))
        print("\n")

print("\nrunOpenVX.py completed - V:"+__version__+"\n")
