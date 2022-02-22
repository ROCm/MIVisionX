FROM centos:centos7

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# install mivisionx base dependencies - Level 1
RUN yum -y update --nogpgcheck && yum -y install --nogpgcheck http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install --nogpgcheck gcc gcc-c++ kernel-devel make cmake3 git && yum-config-manager --enable rhel-server-rhscl-7-rpms && \
        yum -y install --nogpgcheck centos-release-scl && yum -y install --nogpgcheck devtoolset-7
# Enable Developer Toolset 7
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7" ]
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y install --nogpgcheck rocm-dev
# install OpenCV & FFMPEG - Level 3
RUN yum -y groupinstall 'Development Tools' --nogpgcheck && yum -y install --nogpgcheck gtk2-devel libjpeg-devel libpng-devel libtiff-devel libavc1394 wget unzip && \
        mkdir opencv && cd opencv && wget https://github.com/opencv/opencv/archive/4.5.5.zip && unzip 4.5.5.zip && \
        mkdir build && cd build && \
        cmake3 -DWITH_OPENCL=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_VA_INTEL=OFF -DWITH_OPENCL_SVM=OFF ../opencv-4.5.5 && \
        make -j8 && make install
RUN yum -y install --nogpgcheck autoconf automake bzip2 bzip2-devel cmake freetype-devel gcc gcc-c++ git libtool make pkgconfig zlib-devel && \
        yum -y install --nogpgcheck nasm && yum -y --enablerepo=extras install --nogpgcheck epel-release && yum -y install --nogpgcheck yasm && \
        yum -y install --nogpgcheck libx264-devel libx265-devel && \
        yum -y install --nogpgcheck https://forensics.cert.org/cert-forensics-tools-release-el7.rpm && yum -y --enablerepo=forensics install --nogpgcheck fdk-aac && \
        yum -y install --nogpgcheck libass-devel && \
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/" && \
        wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip && unzip n4.0.4.zip && cd FFmpeg-n4.0.4/ && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && make install
# install MIVisionX neural net dependency - Level 4
RUN yum -y install --nogpgcheck libsqlite3x-devel python-devel python3-devel python3-venv bzip2-devel openssl-devel autoconf automake libtool curl make g++ unzip && \
        mkdir neuralNet && cd neuralNet && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && cp half-files/include/half.hpp /usr/local/include/ && \
        wget https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.bz2 && tar xjvf boost_1_72_0.tar.bz2 && \
        cd boost_1_72_0 && ./bootstrap.sh --prefix=/usr/local --with-python=python3.6.8 && \
        ./b2 stage -j16 threading=multi link=shared cxxflags="-std=c++11" && \
        ./b2 install threading=multi link=shared --with-system --with-filesystem && \
        ./b2 stage -j16 threading=multi link=static cxxflags="-std=c++11 -fpic" cflags="-fpic" && \
        ./b2 install threading=multi link=static --with-system --with-filesystem && cd ../ && \
        git clone -b v3.12.0 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && make install
RUN git clone -b rocm-4.2.0 https://github.com/RadeonOpenCompute/rocm-cmake.git && cd rocm-cmake && mkdir build && cd build && \
        cmake3 ../ && make -j8 && make install && cd ../../ && \
        wget https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/archive/1.1.5.zip && unzip 1.1.5.zip && \
        cd MIOpenGEMM-1.1.5 && mkdir build && cd build && cmake3 ../ && make -j8 && make install && cd ../../ && \
        yum -y install --nogpgcheck miopen-opencl

WORKDIR /workspace

# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake3 ../MIVisionX && make -j8 && make install