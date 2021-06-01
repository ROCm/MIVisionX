# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
__copyright__ = "Copyright 2018 - 2021, AMD MIVisionX - Neural Net Test Full Report"
__license__ = "MIT"
__version__ = "1.1.1"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiriti.NageshGowda@amd.com"
__status__ = "Shipping"

if sys.version_info[0] < 3:
    import commands
else:
    import subprocess


def shell(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output = p.communicate()[0][0:-1]
    return output


def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


def write_lines_as_table(header, lines, f):
    for h in header:
        f.write("|%s" % h)
    f.write("|\n")

    for h in header:
        f.write("|:---")
    f.write("|\n")

    for l in lines:
        fields = l.split()
        for field in fields:
            f.write("|%s" % field)
        f.write("|\n")


def strip_libtree_addresses(lib_tree):
    return lib_tree


def script_info():
    print("\nMIVisionX runNeuralNetworkTests V-"+__version__+"\n")
    print(
        "--profiler_mode       - NN Profile Mode: optional (default:0 [range:0 - 9])")
    print("    --profiler_mode 0 -- Run All Tests")
    print("    --profiler_mode 1 -- Run caffe2nnir2openvx No Fuse flow")
    print("    --profiler_mode 2 -- Run caffe2nnir2openvx Fuse flow")
    print("    --profiler_mode 3 -- Run caffe2nnir2openvx FP16 flow")
    print("    --profiler_mode 4 -- Run onnx2nnir2openvx No Fuse flow")
    print("    --profiler_mode 5 -- Run onnx2nnir2openvx Fuse flow")
    print("    --profiler_mode 6 -- Run onnx2nnir2openvx FP16 flow")
    print("    --profiler_mode 7 -- Run nnef2nnir2openvx No Fuse flow")
    print("    --profiler_mode 8 -- Run nnef2nnir2openvx Fuse flow")
    print("    --profiler_mode 9 -- Run nnef2nnir2openvx FP16 flow")
    print(
        "--profiler_level      - NN Profile Batch Size in powers of 2: optional (default:7 [range:1 - N])")
    print(
        "--miopen_find         - MIOPEN_FIND_ENFORCE mode: optional (default:1 [range:1 - 5])")


# models to run - add new models `modelname` , c, h, w
caffeModelConfig = [
    ('caffe-mnist', 1, 28, 28)
]

onnxModelConfig = [
    ('onnx-squeezenet', 3, 224, 224)
]

nnefModelConfig = [
    ('nnef-mnist', 1, 28, 28)
]

# REPORT
reportConfig = [
    ('CAFFE no fused OPs', 'caffe2nnir2openvx_noFuse_profile.md'),
    ('CAFFE fused OPs', 'caffe2nnir2openvx_Fuse_profile.md'),
    ('CAFFE fp16', 'caffe2nnir2openvx_FP16_profile.md'),
    ('ONNX no fused OPs', 'onnx2nnir2openvx_noFuse_profile.md'),
    ('ONNX fused Ops', 'onnx2nnir2openvx_Fuse_profile.md'),
    ('ONNX fp16', 'onnx2nnir2openvx_FP16_profile.md'),
    ('NNEF no fused OPs', 'nnef2nnir2openvx_noFuse_profile.md'),
    ('NNEF fused OPs', 'nnef2nnir2openvx_Fuse_profile.md'),
    ('NNEF fp16', 'nnef2nnir2openvx_FP16_profile.md')
]

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--profiler_mode',      type=int, default=0,
                    help='NN Profile Mode - optional (default:0 [range:0 - 9])')
parser.add_argument('--profiler_level',     type=int, default=7,
                    help='NN Profile Batch Size in powers of 2 - optional (default:7 [range:1 - N])')
parser.add_argument('--miopen_find',        type=int, default=1,
                    help='MIOPEN_FIND_ENFORCE mode - optional (default:1 [range:1 - 5])')
parser.add_argument('--test_info',          type=str, default='no',
                    help='Show test info - optional (default:no [options:no/yes])')
args = parser.parse_args()

profileMode = args.profiler_mode
profileLevel = args.profiler_level
miopenFind = args.miopen_find
testInfo = args.test_info

platfromInfo = platform.platform()

# check arguments
if not 0 <= profileMode <= 9:
    print(
        "\nERROR: NN Profile Mode not in range - [0 - 9]\n")
    exit()
if not 1 <= profileLevel <= 10:
    print(
        "\nERROR: NN Profile Batch Size in powers of 2 not in range - [1 - 10]\n")
    exit()
if not 1 <= miopenFind <= 5:
    print(
        "\nERROR: MIOPEN_FIND_ENFORCE not in range - [1 - 5]\n")
    exit()
if testInfo not in ('no', 'yes'):
    print("ERROR: Show test info options supported - [no or yes]")
    script_info()
    exit()

if testInfo == 'yes':
    script_info()
    exit()

print("\nMIVisionX runNeuralNetworkTests V-"+__version__+"\n")

# check for Scripts
scriptPath = os.path.dirname(os.path.realpath(__file__))
modelCompilerDir = os.path.expanduser(
    '/opt/rocm/mivisionx/model_compiler/python')
pythonScript = modelCompilerDir+'/caffe_to_nnir.py'
modelCompilerScript = os.path.abspath(pythonScript)
if(os.path.isfile(modelCompilerScript)):
    print("\nMIVisionX Neural Net Tests on "+platfromInfo+"\n")
    print("STATUS: Model Compiler Scripts Used from - "+modelCompilerDir+"\n")
else:
    print("ERROR: Model Compiler Scripts Not Found at - "+modelCompilerDir)
    print("ERROR: MIVisionX Not Installed, install MIVisionX and rerun")
    exit()

# Install Model Compiler Deps
modelCompilerDeps = os.path.expanduser('~/.mivisionx-model-compiler-deps')
linuxCMake = 'cmake'
if not os.path.exists(modelCompilerDeps):
    print("STATUS: Model Compiler Deps Install - "+modelCompilerDeps+"\n")
    # sudo requirement check
    sudoLocation = ''
    userName = ''
    if sys.version_info[0] < 3:
        status, sudoLocation = commands.getstatusoutput("which sudo")
        if sudoLocation != '/usr/bin/sudo':
            status, userName = commands.getstatusoutput("whoami")
    else:
        status, sudoLocation = subprocess.getstatusoutput("which sudo")
        if sudoLocation != '/usr/bin/sudo':
            status, userName = subprocess.getstatusoutput("whoami")

    linuxSystemInstall = ''
    linuxSystemInstall_check = ''
    if "centos" in platfromInfo or "redhat" in platfromInfo:
        linuxSystemInstall = 'yum -y'
        linuxSystemInstall_check = '--nogpgcheck'
        if "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
            linuxCMake = 'cmake3'
            os.system(linuxSystemInstall+' ' +
                      linuxSystemInstall_check+' install cmake3')
    elif "Ubuntu" in platfromInfo:
        linuxSystemInstall = 'apt-get -y'
        linuxSystemInstall_check = '--allow-unauthenticated'

    if userName == 'root':
        os.system(linuxSystemInstall+' update')
        os.system(linuxSystemInstall+' install sudo')

    os.makedirs(modelCompilerDeps)
    os.system('sudo -v')
    if "Ubuntu" in platfromInfo:
        os.system(
            'sudo '+linuxSystemInstall+' ' +
            linuxSystemInstall_check+' install git inxi python3 python3-pip protobuf-compiler libprotoc-dev')
    elif "centos" in platfromInfo or "redhat" in platfromInfo:
        os.system(
            'sudo '+linuxSystemInstall+' ' +
            linuxSystemInstall_check+' install git inxi python3-devel python3-pip protobuf python3-protobuf')
    os.system('pip3 install future pytz numpy')
    # Install CAFFE Deps
    os.system('pip3 install google protobuf')
    # Install ONNX Deps
    os.system('pip3 install onnx')
    # Install NNEF Deps
    os.system('mkdir -p '+modelCompilerDeps+'/nnef-deps')
    os.system(
        '(cd '+modelCompilerDeps+'/nnef-deps; git clone https://github.com/KhronosGroup/NNEF-Tools.git)')
    os.system(
        '(cd '+modelCompilerDeps+'/nnef-deps/NNEF-Tools/parser/cpp; mkdir -p build && cd build; '+linuxCMake+' ..; make)')
    os.system(
        '(cd '+modelCompilerDeps+'/nnef-deps/NNEF-Tools/parser/python; sudo python3 setup.py install)')
else:
    print("STATUS: Model Compiler Deps Pre-Installed - "+modelCompilerDeps+"\n")
    if "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
        linuxCMake = 'cmake3'


# Create working directory
outputDirectory = scriptPath+'/models/develop'
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
else:
    shutil.rmtree(outputDirectory)
    os.makedirs(outputDirectory)

# run caffe2nnir2openvx no fuse flow
if profileMode == 0 or profileMode == 1:
    outputDirectory = scriptPath+'/models/develop/caffeNoFuse'
    os.makedirs(outputDirectory)
    for i in range(len(caffeModelConfig)):
        modelName, channel, height, width = caffeModelConfig[i]
        print("\n caffe2nnir2openvx with NO FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/caffe_to_nnir.py '+scriptPath+'/models/' +
                      modelName+'/model.caffemodel . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --fuse-ops 0 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/caffe_no_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/caffe_no_fuse_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_no_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_noFuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_no_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_noFuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/caffe2nnir2openvx_noFuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_no_fuse_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_noFuse_profile.md'''
    os.system(runAwk_md)

# run caffe2nnir2openvx with fuse flow
if profileMode == 0 or profileMode == 2:
    outputDirectory = scriptPath+'/models/develop/caffeFuse'
    os.makedirs(outputDirectory)
    for i in range(len(caffeModelConfig)):
        modelName, channel, height, width = caffeModelConfig[i]
        print("\n caffe2nnir2openvx with FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/caffe_to_nnir.py '+scriptPath+'/models/' +
                      modelName+'/model.caffemodel . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --fuse-ops 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/caffe_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/caffe_fuse_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_Fuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_Fuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/caffe2nnir2openvx_Fuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_fuse_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_Fuse_profile.md'''
    os.system(runAwk_md)

# run caffe2nnir2openvx with fp16 flow
if profileMode == 0 or profileMode == 3:
    outputDirectory = scriptPath+'/models/develop/caffeFP16'
    os.makedirs(outputDirectory)
    for i in range(len(caffeModelConfig)):
        modelName, channel, height, width = caffeModelConfig[i]
        print("\n caffe2nnir2openvx FP16 -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/caffe_to_nnir.py '+scriptPath+'/models/' +
                      modelName+'/model.caffemodel . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --convert-fp16 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/caffe_fp16_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/caffe_fp16_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_fp16_output.log > ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_FP16_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_fp16_output.log > ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_FP16_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/caffe2nnir2openvx_FP16_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/caffe_fp16_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/caffe2nnir2openvx_FP16_profile.md'''
    os.system(runAwk_md)

# run onnx2nnir2openvx no fuse flow
if profileMode == 0 or profileMode == 4:
    outputDirectory = scriptPath+'/models/develop/onnxNoFuse'
    os.makedirs(outputDirectory)
    for i in range(len(onnxModelConfig)):
        modelName, channel, height, width = onnxModelConfig[i]
        print("\n onnx2nnir2openvx with NO FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/onnx_to_nnir.py '+scriptPath+'/models/' +
                      modelName+'/model.onnx . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --fuse-ops 0 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/onnx_no_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/onnx_no_fuse_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_no_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_noFuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_no_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_noFuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/onnx2nnir2openvx_noFuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_no_fuse_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_noFuse_profile.md'''
    os.system(runAwk_md)

# run onnx2nnir2openvx with fuse flow
if profileMode == 0 or profileMode == 5:
    outputDirectory = scriptPath+'/models/develop/onnxFuse'
    os.makedirs(outputDirectory)
    for i in range(len(onnxModelConfig)):
        modelName, channel, height, width = onnxModelConfig[i]
        print("\n onnx2nnir2openvx with FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/onnx_to_nnir.py '+scriptPath+'/models/' +
                      modelName+'/model.onnx . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --fuse-ops 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/onnx_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/onnx_fuse_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_Fuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_Fuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/onnx2nnir2openvx_Fuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_fuse_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_Fuse_profile.md'''
    os.system(runAwk_md)

# run onnx2nnir2openvx with fp16 flow
if profileMode == 0 or profileMode == 6:
    outputDirectory = scriptPath+'/models/develop/onnxFP16'
    os.makedirs(outputDirectory)
    for i in range(len(onnxModelConfig)):
        modelName, channel, height, width = onnxModelConfig[i]
        print("\n onnx2nnir2openvx FP16 -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/onnx_to_nnir.py '+scriptPath+'/models/' +
                      modelName+'/model.onnx . --input-dims '+x+','+str(channel)+','+str(height)+','+str(width)+')')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --convert-fp16 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/onnx_fp16_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/onnx_fp16_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_fp16_output.log > ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_FP16_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_fp16_output.log > ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_FP16_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/onnx2nnir2openvx_FP16_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/onnx_fp16_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/onnx2nnir2openvx_FP16_profile.md'''
    os.system(runAwk_md)

# run nnef2nnir2openvx no fuse flow
if profileMode == 0 or profileMode == 7:
    outputDirectory = scriptPath+'/models/develop/nnefNoFuse'
    os.makedirs(outputDirectory)
    for i in range(len(nnefModelConfig)):
        modelName, channel, height, width = nnefModelConfig[i]
        print("\n nnef2nnir2openvx with No FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/nnef_to_nnir.py '+scriptPath+'/models/' +
                      modelName+' . )')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --fuse-ops 0 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/nnef_no_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/nnef_no_fuse_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_no_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_noFuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_no_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_noFuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/nnef2nnir2openvx_noFuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_no_fuse_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_noFuse_profile.md'''
    os.system(runAwk_md)

# run nnef2nnir2openvx fuse flow
if profileMode == 0 or profileMode == 8:
    outputDirectory = scriptPath+'/models/develop/nnefFuse'
    os.makedirs(outputDirectory)
    for i in range(len(nnefModelConfig)):
        modelName, channel, height, width = nnefModelConfig[i]
        print("\n nnef2nnir2openvx with FUSED Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/nnef_to_nnir.py '+scriptPath+'/models/' +
                      modelName+' . )')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --fuse-ops 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/nnef_fuse_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/nnef_fuse_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_Fuse_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_fuse_output.log > ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_Fuse_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/nnef2nnir2openvx_Fuse_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_fuse_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_Fuse_profile.md'''
    os.system(runAwk_md)

# run nnef2nnir2openvx FP16 flow
if profileMode == 0 or profileMode == 9:
    outputDirectory = scriptPath+'/models/develop/nnefFP16'
    os.makedirs(outputDirectory)
    for i in range(len(nnefModelConfig)):
        modelName, channel, height, width = nnefModelConfig[i]
        print("\n nnef2nnir2openvx with FP16 Operations -- "+modelName+"\n")
        modelBuildDir = outputDirectory+'/nnir_build_'
        for x in range(profileLevel):
            x = 2**x
            x = str(x)
            print("\n"+modelName+" - Batch size "+x)
            os.system('(cd '+outputDirectory +
                      '; mkdir -p nnir_build_'+x+')')
            os.system('(cd '+modelBuildDir+x+'; python3 '+modelCompilerDir+'/nnef_to_nnir.py '+scriptPath+'/models/' +
                      modelName+' . )')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_update.py --convert-fp16 1 . .)')
            os.system('(cd '+modelBuildDir+x+'; python3 ' +
                      modelCompilerDir+'/nnir_to_openvx.py . .)')
            os.system('(cd '+modelBuildDir+x+'; '+linuxCMake+' .; make)')
            os.system('echo '+modelName+' - Batch size '+x+'  | tee -a ' +
                      scriptPath+'/models/develop/nnef_fp16_output.log')
            os.system('(cd '+modelBuildDir+x+'; MIOPEN_FIND_ENFORCE='+str(miopenFind) +
                      ' ./anntest weights.bin | tee -a '+scriptPath+'/models/develop/nnef_fp16_output.log)')

    runAwk_csv = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s,%3d,%8.3f ms,%8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_fp16_output.log > ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_FP16_profile.csv'''
    os.system(runAwk_csv)
    runAwk_txt = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("%-16s %3d %8.3f ms %8.3f ms\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_fp16_output.log > ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_FP16_profile.txt'''
    os.system(runAwk_txt)

    orig_stdout = sys.stdout
    sys.stdout = open(
        scriptPath+'/models/develop/nnef2nnir2openvx_FP16_profile.md', 'a')
    echo_1 = '|      Model Name      | Batch Size | Time/Batch (ms) | Time/Image (ms) |'
    print(echo_1)
    echo_2 = '|----------------------|------------|-----------------|-----------------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    runAwk_md = r'''awk 'BEGIN { net = "xxx"; bsize = 1; } / - Batch size/ { net = $1; bsize = $5; } /average over 100 iterations/ { printf("|%-22s|%-12d|%-17.3f|%-17.3f|\n", net, bsize, $4, $4/bsize) }' ''' + \
        scriptPath+'''/models/develop/nnef_fp16_output.log | tee -a ''' + \
        scriptPath+'''/models/develop/nnef2nnir2openvx_FP16_profile.md'''
    os.system(runAwk_md)

# get system data
platform_name = platform.platform()
platform_name_fq = shell('hostname --all-fqdns')
platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

file_dtstr = datetime.now().strftime("%Y%m%d")
reportFilename = 'inference_report_%s_%s.md' % (
    platform_name, file_dtstr)
report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
sys_info = shell('inxi -c0 -S')

cpu_info = shell('inxi -c0 -C')
cpu_info = cpu_info.rstrip()  # strip out clock speeds

gpu_info = shell('inxi -c0 -G')
gpu_info = gpu_info.rstrip()  # strip out X info

memory_info = shell('inxi -c 0 -m')
board_info = shell('inxi -c0 -M')

lib_tree = shell('ldd -v /opt/rocm/mivisionx/lib/libvx_nn.so')
lib_tree = strip_libtree_addresses(lib_tree)

vbios = shell('(cd /opt/rocm/bin/; ./rocm-smi -v)')

rocmInfo = shell('(cd /opt/rocm/bin/; ./rocm-smi -a)')

rocm_packages = shell('dpkg-query -W | grep rocm')
rocm_packages = rocm_packages.splitlines()

# Write Report
with open(reportFilename, 'w') as f:
    f.write("MIVisionX - ML Inference Report\n")
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

    if profileMode == 0:
        for i in range(len(reportConfig)):
            modelType, reportFile = reportConfig[i]
            f.write("\n### MODEL FORMAT: %s\n" % modelType)
            with open(scriptPath+'/models/develop/'+reportFile) as benchmarkFile:
                for line in benchmarkFile:
                    f.write("%s" % line)
    else:
        modelType, reportFile = reportConfig[profileMode - 1]
        f.write("\n### MODEL FORMAT: %s\n" % modelType)
        with open(scriptPath+'/models/develop/'+reportFile) as benchmarkFile:
            for line in benchmarkFile:
                f.write("%s" % line)

    f.write("\n")

    f.write("ROCm Package and Version Report\n")
    f.write("-------------\n")
    f.write("\n")
    write_lines_as_table(['Package', 'Version'], rocm_packages, f)
    f.write("\n\n\n")

    f.write("Vbios Report\n")
    f.write("-------------\n")
    f.write("\n")
    write_formatted(vbios, f)
    f.write("\n")
    f.write("ROCm Device Info Report\n")
    f.write("-------------\n")
    f.write("\n")
    write_formatted(rocmInfo, f)
    f.write("\n")

    f.write("Dynamic Libraries Report\n")
    f.write("-----------------\n")
    f.write("\n")
    write_formatted(lib_tree, f)
    f.write("\n")

    f.write("\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2021 -- runNeuralNetworkTests.py V-"+__version__+"**\n")

# report file
reportFileDir = os.path.abspath(reportFilename)
print("\nSTATUS: Output Report File - "+reportFileDir)

print("\nrunNeuralNetworkTests.py completed - V:"+__version__+"\n")
