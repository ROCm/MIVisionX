__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2018, AMD Radeon MIVisionX setup"
__license__     = "MIT"
__version__     = "0.9.93"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "Kiriti.NageshGowda@amd.com"
__status__      = "beta"

import argparse
import commands
import os

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='',        help='Setup home directory - optional (default:~/)')
parser.add_argument('--installer', type=str, default='apt-get', help='Linux system installer - optional (default:apt-get) [options: Ubuntu - apt-get; CentOS - yum]')
parser.add_argument('--miopen',    type=str, default='1.7.1',   help='MIOpen Version - optional (default:1.7.1)')
args = parser.parse_args()

setupDir = args.directory
linuxSystemInstall = args.installer
MIOpenVersion = args.miopen

# sudo requirement check
sudoLocation = ''
userName = ''
status, sudoLocation = commands.getstatusoutput("which sudo")
if sudoLocation != '/usr/bin/sudo':
	status, userName = commands.getstatusoutput("whoami")

if setupDir == '':
	setupDir_deps = '~/deps'
else:
	setupDir_deps = setupDir+'/deps'

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
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install cmake3 boost boost-thread boost-devel openssl-devel')

# setup directory
deps_dir = os.path.expanduser(setupDir_deps)

# MIVisionX setup
if(os.path.exists(deps_dir)):
	print("\nMIVisionX Dependencies Installed\n")
else:
	print("\nMIVisionX Dependencies Installation\n")
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install cmake git wget unzip')
	os.system('(cd '+setupDir+'; mkdir deps)')
	os.system('(cd '+setupDir+'; mkdir deps)')
	os.system('(cd '+deps_dir+'; git clone https://github.com/RadeonOpenCompute/rocm-cmake.git )')
	os.system('(cd '+deps_dir+'; git clone https://github.com/ROCmSoftwarePlatform/MIOpenGEMM.git )')
	os.system('(cd '+deps_dir+'; wget https://github.com/ROCmSoftwarePlatform/MIOpen/archive/'+MIOpenVersion+'.zip )')
	os.system('(cd '+deps_dir+'; unzip '+MIOpenVersion+'.zip )')
	os.system('(cd '+deps_dir+'; wget https://github.com/protocolbuffers/protobuf/archive/v3.5.2.zip )')
	os.system('(cd '+deps_dir+'; unzip v3.5.2.zip )')
	os.system('(cd '+deps_dir+'; wget https://github.com/opencv/opencv/archive/3.4.0.zip )')
	os.system('(cd '+deps_dir+'; unzip 3.4.0.zip )')
	os.system('(cd '+deps_dir+'; mkdir build )')
	os.system('(cd '+deps_dir+'/build; mkdir rocm-cmake MIOpenGEMM MIOpen OpenCV )')
	os.system('(cd '+deps_dir+'/build/rocm-cmake; '+linuxCMake+' ../../rocm-cmake )')
	os.system('(cd '+deps_dir+'/build/rocm-cmake; make -j8 )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/rocm-cmake; sudo '+linuxFlag+' make install )')
	os.system('(cd '+deps_dir+'/build/MIOpenGEMM; '+linuxCMake+' ../../MIOpenGEMM )')
	os.system('(cd '+deps_dir+'/build/MIOpenGEMM; make -j8 )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/MIOpenGEMM; sudo '+linuxFlag+' make install )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/MIOpen-'+MIOpenVersion+'; sudo '+linuxFlag+' '+linuxCMake+' -P install_deps.cmake )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/MIOpen; sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install libssl-dev libboost-dev libboost-system-dev libboost-filesystem-dev  )')
	os.system('(cd '+deps_dir+'/build/MIOpen; '+linuxCMake+' -DMIOPEN_BACKEND=OpenCL ../../MIOpen-'+MIOpenVersion+' )')
	os.system('(cd '+deps_dir+'/build/MIOpen; make -j8 )')
	os.system('(cd '+deps_dir+'/build/MIOpen; make MIOpenDriver )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/MIOpen; sudo '+linuxFlag+' make install )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/MIOpen; sudo '+linuxFlag+' '+linuxSystemInstall+' autoremove )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install autoconf automake libtool curl make g++ unzip )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' '+linuxSystemInstall+' autoremove )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; ./autogen.sh )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; ./configure )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; make -j16 )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; make check -j16 )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' make install )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' ldconfig )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install python-pip )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' yes | pip install protobuf )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' yes | pip install pytz )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' yes | pip install numpy )')
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
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install python-matplotlib python-numpy python-pil python-scipy python-skimage cython')
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install qt5-default qtcreator')
