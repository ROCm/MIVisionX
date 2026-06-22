# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys
import argparse
import platform
import traceback
if sys.version_info[0] < 3:
    import commands
else:
    import subprocess

libraryName = "MIVisionX"

__copyright__ = f"Copyright(c) 2018 - 2025, AMD ROCm {libraryName}"
__version__ = "4.0.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# ANSI Escape codes for info messages
TEXT_WARNING = "\033[93m\033[1m"
TEXT_ERROR = "\033[91m\033[1m"
TEXT_INFO = "\033[1m"
TEXT_DEFAULT = "\033[0m"

def info(msg):
    print(f"{TEXT_INFO}INFO:{TEXT_DEFAULT} {msg}")

def warn(msg):
    print(f"{TEXT_WARNING}WARNING:{TEXT_DEFAULT} {msg}")

def error(msg):
    print(f"{TEXT_ERROR}ERROR:{TEXT_DEFAULT} {msg}")

# error check for calls
def ERROR_CHECK(waitval):
    if(waitval != 0): # return code and signal flags
        error('ERROR_CHECK failed with status:'+str(waitval))
        traceback.print_stack()
        status = ((waitval >> 8) | waitval) & 255 # combine exit code and wait flags into single non-zero byte
        exit(status)

def install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, package_list):
    cmd_str = 'sudo ' + linuxFlag + ' ' + linuxSystemInstall + \
        ' ' + linuxSystemInstall_check+' install '
    for i in range(len(package_list)):
        cmd_str += package_list[i] + " "
    ERROR_CHECK(os.system(cmd_str))

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--directory', 	type=str, default='~/mivisionx-deps',
                    help='Setup home directory - optional (default:~/)')
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')
parser.add_argument('--opencv',    	type=str, default='4.6.0',
                    help='OpenCV Version - optional, only used by runvx for image/video display (default for non Ubuntu OS:4.6.0)')
parser.add_argument('--developer', 	type=str, default='OFF',
                    help='Setup Developer Options - optional (default:OFF) [options:ON/OFF]')
parser.add_argument('--reinstall', 	type=str, default='OFF',
                    help='Remove previous setup and reinstall - optional (default:OFF) [options:ON/OFF]')
parser.add_argument('--backend', 	type=str, default='HIP',
                    help='MIVisionX Dependency Backend - optional (default:HIP) [options:HIP/CPU/OCL]')
args = parser.parse_args()

setupDir = args.directory
opencvVersion = args.opencv
developerInstall = args.developer.upper()
reinstall = args.reinstall.upper()
backend = args.backend.upper()
ROCM_PATH = args.rocm_path

# check inputs
if developerInstall not in ('OFF', 'ON'):
    error(
        "ERROR: Developer Option Not Supported - [Supported Options: OFF or ON]\n")
    exit(-1)
if reinstall not in ('OFF', 'ON'):
    error(
        "ERROR: Re-Install Option Not Supported - [Supported Options: OFF or ON]\n")
    parser.print_help()
    exit(-1)
if backend not in ('OCL', 'HIP', 'CPU'):
    error(
        "ERROR: Backend Option Not Supported - [Supported Options: CPU or OCL or HIP]\n")
    parser.print_help()
    exit(-1)

# override default path if env path set 
if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
info("ROCm PATH set to -- "+ROCM_PATH+"\n")


# check ROCm installation
if os.path.exists(ROCM_PATH) and backend != 'CPU':
    info("ROCm Installation Found -- "+ROCM_PATH+"\n")
    os.system('echo ROCm Info -- && '+ROCM_PATH+'/bin/rocminfo')
else:
    if backend != 'CPU':
        warn("\nWARNING: ROCm Not Found at -- "+ROCM_PATH+"\n")
        warn(
            "WARNING: If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]\n")
        warn("WARNING: Limited dependencies will be installed\n")
        backend = 'CPU'
    else:
        info("STATUS: CPU Backend Install\n")

# Setup Directory for Deps
if setupDir == '~/mivisionx-deps':
    setupDir_deps = setupDir
else:
    setupDir_deps = setupDir+'/mivisionx-deps'

# setup directory path
deps_dir = os.path.expanduser(setupDir_deps)
deps_dir = os.path.abspath(deps_dir)

# get platform info
platformInfo = platform.platform()

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

# check os version
os_info_data = 'NOT Supported'
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as os_file:
        os_info_data = os_file.read().replace('\n', ' ')
        os_info_data = os_info_data.replace('"', '')

# setup for Linux
linuxSystemInstall = ''
linuxCMake = 'cmake'
linuxSystemInstall_check = ''
linuxFlag = ''
sudoValidate = 'sudo -v'
osUpdate = ''
if "centos" in os_info_data or "redhat" in os_info_data or "Oracle" in os_info_data:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    osUpdate = 'makecache'
    if "VERSION_ID=8" in os_info_data:
        platformInfo = platformInfo+'-centos-8-based'
    elif "VERSION_ID=9" in os_info_data:
        platformInfo = platformInfo+'-centos-9-based'
    else:
        platformInfo = platformInfo+'-centos-undefined-version'
elif "Ubuntu" in os_info_data:
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    osUpdate = 'update'
    linuxFlag = '-S'
    if "VERSION_ID=22" in os_info_data:
        platformInfo = platformInfo+'-ubuntu-22'
    elif "VERSION_ID=24" in os_info_data:
        platformInfo = platformInfo+'-ubuntu-24'
    else:
        platformInfo = platformInfo+'-ubuntu-undefined-version'
elif "SLES" in os_info_data:
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    osUpdate = 'refresh'
    platformInfo = platformInfo+'-sles'
elif "Mariner" in os_info_data:
    linuxSystemInstall = 'tdnf -y'
    linuxSystemInstall_check = '--nogpgcheck'
    platformInfo = platformInfo+'-mariner'
    osUpdate = 'makecache'
else:
    error("MIVisionX Setup on "+platformInfo+" is unsupported\n")
    error("MIVisionX Setup Supported on: Ubuntu 22/24, RedHat 8/9, & SLES 15\n")
    exit(-1)

# MIVisionX Setup
info(f"{libraryName} Setup on: "+platformInfo)
info(f"{libraryName} Dependencies Installation with MIVisionX-setup.py V-"+__version__)

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' '+osUpdate))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# Delete previous install
if reinstall == 'ON':
    ERROR_CHECK(os.system(sudoValidate))
    if os.path.exists(deps_dir):
        ERROR_CHECK(os.system('sudo rm -rf '+deps_dir))
        info("MIVisionX Setup: Removing Previous Install -- "+deps_dir+"\n")

# common packages
coreCommonPackages = [
    'cmake',
    'wget',
    'unzip',
    'pkg-config',
    'inxi'
]

# float16 support for core OpenVX and the RPP extension
halfDebianPackages = [
    'half'
]

halfRPMPackages = [
    'half'
]

rppDebianPackages = [
    'rpp',
    'rpp-dev'
]

rppRPMPackages = [
    'rpp',
    'rpp-devel'
]

openclDebianPackages = [
    'ocl-icd-opencl-dev'
]

openclRPMPackages = [
    'ocl-icd-devel'
]

opencvDebianPackages = [
    'libopencv-dev'
]

opencvRPMPackages = [
    'gtk2-devel',
    'libjpeg-devel',
    'libpng-devel',
    'libtiff-devel',
    'libavc1394'
]

# update
ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +' '+linuxSystemInstall_check+' '+osUpdate))

ERROR_CHECK(os.system(sudoValidate))
# common packages
install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, coreCommonPackages)
# GPU backend support (HIP or OpenCL) -- float16 + RPP extension
if backend in ('HIP', 'OCL'):
    if "ubuntu" in platformInfo:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, halfDebianPackages)
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, rppDebianPackages)
    else:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, halfRPMPackages)
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, rppRPMPackages)

# Install OpenCL ICD Loader
if backend == 'OCL':
    if "ubuntu" in platformInfo:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, openclDebianPackages)
    else:
        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, openclRPMPackages)

# OpenCV
if "ubuntu" in platformInfo:
    install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, opencvDebianPackages)

if os.path.exists(deps_dir):
    info("MIVisionX Setup: Re-Installing Libraries from -- "+deps_dir+"\n")
# Clean Install
else:
    info("MIVisionX Dependencies Clean Installation with MIVisionX-setup.py V-"+__version__+"\n")
    ERROR_CHECK(os.system(sudoValidate))
    # Create deps & build folder
    ERROR_CHECK(os.system('mkdir '+deps_dir))
    ERROR_CHECK(os.system('(cd '+deps_dir+'; mkdir build )'))

    # Install OpenCV (optional - used by runvx for image/video display)
    ERROR_CHECK(os.system('(cd '+deps_dir+'/build; mkdir OpenCV )'))
    # Install
    if "ubuntu" in platformInfo:
        info("STATUS: MIVisionX Setup: OpenCV Package install supported for Ubuntu\n")
    else:
        if "centos" in platformInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' groupinstall \'Development Tools\''))
        elif "sles" in platformInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install -t pattern devel_basis'))

        install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, opencvRPMPackages)
        # OpenCV 4.6.0
        # Get Source and install
        ERROR_CHECK(os.system(
            '(cd '+deps_dir+'; wget https://github.com/opencv/opencv/archive/'+opencvVersion+'.zip )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'; unzip '+opencvVersion+'.zip )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; '+linuxCMake +
                        ' -D WITH_EIGEN=OFF \
                        -D WITH_GTK=ON \
                        -D WITH_JPEG=ON \
                        -D BUILD_JPEG=ON \
                        -D WITH_OPENCL=OFF \
                        -D WITH_OPENCLAMDFFT=OFF \
                        -D WITH_OPENCLAMDBLAS=OFF \
                        -D WITH_VA_INTEL=OFF \
                        -D WITH_OPENCL_SVM=OFF  \
                        -D CMAKE_INSTALL_PREFIX=/usr/local \
                        -D BUILD_LIST=core,features2d,highgui,imgcodecs,imgproc,photo,video,videoio  \
                        -D CMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
                        ../../opencv-'+opencvVersion+' )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; make -j$(nproc))'))
        ERROR_CHECK(os.system(sudoValidate))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo make install)'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'/build/OpenCV; sudo ldconfig)'))

    if developerInstall == 'ON':
        ERROR_CHECK(os.system(sudoValidate))
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                ' install autoconf texinfo wget'))
        if "ubuntu" in platformInfo:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' install build-essential libgmp-dev'))
        else:
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' install gmp-devel'))
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
                    ' groupinstall \'Development Tools\' '))
        ERROR_CHECK(os.system(
            '(cd '+deps_dir+'; wget https://ftp.gnu.org/gnu/gdb/gdb-12.1.tar.gz )'))
        ERROR_CHECK(os.system('(cd '+deps_dir+'; tar -xvzf gdb-12.1.tar.gz )'))
        ERROR_CHECK(os.system(
            '(cd '+deps_dir+'/gdb-12.1; ./configure --with-python3; make CXXFLAGS="-static-libstdc++" -j$(nproc); sudo make install)'))

info(f"{libraryName} Dependencies Installed with MIVisionX-setup.py V-"+__version__+" on "+platformInfo+"\n")