__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2018, AMD Radeon MIVisionX setup"
__license__     = "MIT"
__version__     = "1.3.0"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "Kiriti.NageshGowda@amd.com"
__status__      = "Shipping"

import argparse
import commands
import os

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='',        help='Setup home directory - optional (default:~/)')
parser.add_argument('--installer', type=str, default='apt-get', help='Linux system installer - optional (default:apt-get) [options: Ubuntu - apt-get; CentOS - yum]')
parser.add_argument('--miopen',    type=str, default='1.8.1',   help='MIOpen Version - optional (default:1.8.1)')
parser.add_argument('--ffmpeg',    type=str, default='no',      help='FFMPEG Installation - optional (default:no) [options: Install ffmpeg - yes')
args = parser.parse_args()

setupDir = args.directory
linuxSystemInstall = args.installer
MIOpenVersion = args.miopen
ffmpegInstall = args.ffmpeg

# sudo requirement check
sudoLocation = ''
userName = ''
status, sudoLocation = commands.getstatusoutput("which sudo")
if sudoLocation != '/usr/bin/sudo':
	status, userName = commands.getstatusoutput("whoami")

if setupDir == '':
	setupDir_deps = '~/mivisionx-deps'
else:
	setupDir_deps = setupDir+'/mivisionx-deps'

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
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install cmake3 boost boost-thread boost-devel openssl-devel hg')

# setup directory
deps_dir = os.path.expanduser(setupDir_deps)

# MIVisionX setup
if(os.path.exists(deps_dir)):
	print("\nMIVisionX Dependencies Installed\n")
else:
	print("\nMIVisionX Dependencies Installation\n")
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install cmake git wget unzip')
	os.system('(cd '+setupDir+'; mkdir mivisionx-deps)')
	# Get Installation Source
	os.system('(cd '+deps_dir+'; git clone https://github.com/RadeonOpenCompute/rocm-cmake.git )')
	os.system('(cd '+deps_dir+'; git clone https://github.com/ROCmSoftwarePlatform/MIOpenGEMM.git )')
	os.system('(cd '+deps_dir+'; git clone --recursive -b n4.0.4 https://git.ffmpeg.org/ffmpeg.git )')
	os.system('(cd '+deps_dir+'; wget https://github.com/ROCmSoftwarePlatform/MIOpen/archive/'+MIOpenVersion+'.zip )')
	os.system('(cd '+deps_dir+'; unzip '+MIOpenVersion+'.zip )')
	os.system('(cd '+deps_dir+'; wget https://github.com/protocolbuffers/protobuf/archive/v3.5.2.zip )')
	os.system('(cd '+deps_dir+'; unzip v3.5.2.zip )')
	os.system('(cd '+deps_dir+'; wget https://github.com/opencv/opencv/archive/3.4.0.zip )')
	os.system('(cd '+deps_dir+'; unzip 3.4.0.zip )')
	os.system('(cd '+deps_dir+'; mkdir build )')
	# Install ROCm-CMake
	os.system('(cd '+deps_dir+'/build; mkdir rocm-cmake MIOpenGEMM MIOpen OpenCV )')
	os.system('(cd '+deps_dir+'/build/rocm-cmake; '+linuxCMake+' ../../rocm-cmake )')
	os.system('(cd '+deps_dir+'/build/rocm-cmake; make -j8 )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/rocm-cmake; sudo '+linuxFlag+' make install )')
	# Install MIOpenGEMM
	os.system('(cd '+deps_dir+'/build/MIOpenGEMM; '+linuxCMake+' ../../MIOpenGEMM )')
	os.system('(cd '+deps_dir+'/build/MIOpenGEMM; make -j8 )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/build/MIOpenGEMM; sudo '+linuxFlag+' make install )')
	os.system('sudo -v')
	# Install MIOpen
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
	# Install ProtoBuf
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install autoconf automake libtool curl make g++ unzip )')
	os.system('sudo -v')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; sudo '+linuxFlag+' '+linuxSystemInstall+' autoremove )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; ./autogen.sh )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; ./configure )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; make -j8 )')
	os.system('(cd '+deps_dir+'/protobuf-3.5.2; make check -j8 )')
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
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install python-matplotlib python-numpy python-pil python-scipy python-skimage cython')
	os.system('sudo -v')
	os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install qt5-default qtcreator')
	# Install ffmpeg
	if ffmpegInstall == 'yes':
		if linuxSystemInstall == 'apt-get':
			os.system('sudo -v')
			os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev')
			os.system('sudo -v')
			os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev')
			os.system('sudo -v')
			os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev')
			os.system('sudo -v')
			os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install nasm yasm libx264-dev libx265-dev libnuma-dev libfdk-aac-dev')
		else:
			os.system('sudo -v')
			os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install autoconf automake bzip2 bzip2-devel cmake freetype-devel libass-devel')
			os.system('sudo -v')
			os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' -y '+linuxSystemInstall_check+' install gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel')
			# Nasm
			os.system('(cd '+deps_dir+'; curl -O -L https://www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.bz2 )')
			os.system('(cd '+deps_dir+'; tar xjvf nasm-2.14.02.tar.bz2 )')
			os.system('(cd '+deps_dir+'/nasm-2.14.02; ./autogen.sh; ./configure; make -j8 )')
			os.system('sudo -v')
			os.system('(cd '+deps_dir+'/nasm-2.14.02; sudo '+linuxFlag+' make install )')
			# Yasm
			os.system('(cd '+deps_dir+'; curl -O -L https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz )')
			os.system('(cd '+deps_dir+'; tar xzvf yasm-1.3.0.tar.gz )')
			os.system('(cd '+deps_dir+'/yasm-1.3.0; ./configure; make -j8 )')
			os.system('sudo -v')
			os.system('(cd '+deps_dir+'/yasm-1.3.0; sudo '+linuxFlag+' make install )')
			# libx264
			os.system('(cd '+deps_dir+'; git clone --depth 1 https://code.videolan.org/videolan/x264.git )')
			os.system('(cd '+deps_dir+'/x264; ./configure --enable-static --disable-opencl; make -j8 )')
			os.system('sudo -v')
			os.system('(cd '+deps_dir+'/x264; sudo '+linuxFlag+' make install )')
			# libx265
			os.system('(cd '+deps_dir+'; hg clone https://bitbucket.org/multicoreware/x265 )')
			os.system('(cd '+deps_dir+'/x265/build/linux; cmake -G "Unix Makefiles" -DENABLE_SHARED:bool=off ../../source; make -j8 )')
			os.system('sudo -v')
			os.system('(cd '+deps_dir+'/x265/build/linux; sudo '+linuxFlag+' make install )')
			# libfdk_aac
			os.system('(cd '+deps_dir+'; git clone --depth 1 https://github.com/mstorsjo/fdk-aac )')
			os.system('(cd '+deps_dir+'/fdk-aac; autoreconf -fiv; ./configure; make -j8 )')
			os.system('sudo -v')
			os.system('(cd '+deps_dir+'/fdk-aac; sudo '+linuxFlag+' make install )')
		os.system('sudo -v')
		os.system('(cd '+deps_dir+'/ffmpeg; sudo '+linuxFlag+' ldconfig )')
		os.system('(cd '+deps_dir+'/ffmpeg; ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree)')
		os.system('(cd '+deps_dir+'/ffmpeg; make -j8 )')
		os.system('sudo -v')
		os.system('(cd '+deps_dir+'/ffmpeg; sudo '+linuxFlag+' make install )')
