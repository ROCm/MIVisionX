FROM centos:centos7

# install mivisionx base dependencies - Level 1
RUN yum -y update && yum -y install http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install gcc gcc-c++ kernel-devel make && yum -y install cmake3 && yum -y install git
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms && yum -y install centos-release-scl && yum -y install devtoolset-7 && \
        echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y install rocm-dev
# Enable Developer Toolset 7
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7" ]
# install OpenCV & FFMPEG - Level 3
RUN yum -y groupinstall 'Development Tools' && yum -y install gtk2-devel libjpeg-devel libpng-devel libtiff-devel libavc1394 wget unzip && \
        mkdir opencv && cd opencv && wget https://github.com/opencv/opencv/archive/3.4.0.zip && unzip 3.4.0.zip && \
        mkdir build && cd build && \
        cmake3 -DWITH_OPENCL=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_VA_INTEL=OFF -DWITH_OPENCL_SVM=OFF ../opencv-3.4.0 && \
        make -j8 && make install
RUN yum -y install autoconf automake bzip2 bzip2-devel cmake freetype-devel gcc gcc-c++ git libtool make pkgconfig zlib-devel && \
        yum -y install nasm && yum -y --enablerepo=extras install epel-release && yum -y install yasm && \
        yum -y install libx264-devel libx265-devel && \
        yum -y install https://forensics.cert.org/cert-forensics-tools-release-el7.rpm && yum -y --enablerepo=forensics install fdk-aac && \
        yum -y install libass-devel && \
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/" && \
        git clone --recursive -b n4.0.4 https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && make install