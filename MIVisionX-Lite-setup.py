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

__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2018, AMD Radeon MIVisionX Lite setup"
__license__     = "MIT"
__version__     = "1.0.0"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "Kiriti.NageshGowda@amd.com"
__status__      = "Shipping"

import argparse
import sys
if sys.version_info[0] < 3:
	import commands
else:
	import subprocess
import os

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='',        help='Setup home directory - optional (default:~/)')
parser.add_argument('--installer', type=str, default='apt-get', help='Linux system installer - optional (default:apt-get) [options: Ubuntu - apt-get; CentOS - yum]')
parser.add_argument('--reinstall', type=str, default='no',      help='Remove previous setup and reinstall - optional (default:no) [options:yes/no]')
args = parser.parse_args()

setupDir = args.directory
linuxSystemInstall = args.installer
reinstall = args.reinstall

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
	

if setupDir == '':
	setupDir_deps = '~/mivisionx-lite-deps'
else:
	setupDir_deps = setupDir+'/mivisionx-lite-deps'

# setup for CentOS or Ubuntu
linuxSystemInstall_check = '--nogpgcheck'
linuxCMake = 'cmake3'
linuxFlag = ''
if linuxSystemInstall == '' or linuxSystemInstall == 'apt-get':
	linuxSystemInstall = 'apt-get'
	linuxSystemInstall_check = '--allow-unauthenticated'
	linuxCMake = 'cmake'
	linuxFlag = '-S'
	if userName == 'root':
		os.system('apt -y update')
		os.system('apt -y install sudo')
else:
	if userName == 'root':
		os.system('yum -y update')
		os.system('yum -y install sudo')
	os.system('sudo -v')
	os.system('sudo yum -y update')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install cmake3 boost boost-thread boost-devel')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install openssl-devel hg autoconf automake')

# setup directory
deps_dir = os.path.expanduser(setupDir_deps)

# Delete previous install
if(os.path.exists(deps_dir) and reinstall == 'yes'):
	os.system('sudo -v')
	os.system('sudo rm -rf '+deps_dir)

# MIVisionX setup
if(os.path.exists(deps_dir)):
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'; sudo cp half-files/include/half.hpp /usr/local/include/ )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/OpenCV; sudo '+linuxFlag+' make install -j8)')
	print("\nMIVisionX Lite Dependencies Installed\n")
else:
	print("\nMIVisionX Lite Dependencies Installation\n")
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install cmake git wget unzip')
	os.system('(cd '+setupDir+'; mkdir mivisionx-lite-deps)')
	# Get Installation Source
	os.system('(cd '+deps_dir+'; wget https://github.com/opencv/opencv/archive/3.4.0.zip )')
	os.system('(cd '+deps_dir+'; unzip 3.4.0.zip )')
	os.system('(cd '+deps_dir+'; mkdir build )')
	# Install half.hpp
	os.system('(cd '+deps_dir+'; wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip )')
	os.system('(cd '+deps_dir+'; unzip half-1.12.0.zip -d half-files )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'; sudo cp half-files/include/half.hpp /usr/local/include/ )')
	# Install OpenCV
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev')
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev')
	os.system('(cd '+deps_dir+'/build/OpenCV; '+linuxCMake+' -DWITH_OPENCL=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_VA_INTEL=OFF -DWITH_OPENCL_SVM=OFF ../../opencv-3.4.0 )')
	os.system('(cd '+deps_dir+'/build/OpenCV; make -j8 )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/OpenCV; sudo '+linuxFlag+' make install )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/OpenCV; sudo '+linuxFlag+' ldconfig )')
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install inxi aha libboost-python-dev build-essential')
	print("\nMIVisionX Lite Dependencies Installed\n")
