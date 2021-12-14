FROM rocm/pytorch:latest

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies 
RUN apt-get -y install gcc g++ cmake git
# install OpenCV & FFMPEG 
RUN apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/3.4.0.zip && unzip 3.4.0.zip && \
        mkdir build && cd build && cmake -DWITH_OPENCL=OFF ../opencv-3.4.0 && make -j8 && sudo make install && sudo ldconfig && cd
RUN apt-get -y install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev \
        libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev \
        nasm yasm libx264-dev libx265-dev libnuma-dev libfdk-aac-dev && \
        wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip && unzip n4.0.4.zip && cd FFmpeg-n4.0.4/ && sudo ldconfig && \
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/" && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && sudo make install && cd
# install MIVisionX rocAL dependency
RUN apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev clang && \
        apt-get -y install libbz2-dev libssl-dev python-dev python3-dev autoconf automake libtool curl make g++ unzip && \
        wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo cp half-files/include/half.hpp /usr/local/include/ && \
        git clone -b v3.12.0 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd && \
        wget https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.bz2 && tar xjvf boost_1_72_0.tar.bz2 && \
        export CPLUS_INCLUDE_PATH=/usr/include/python3.6 && cd boost_1_72_0 && \
        ./bootstrap.sh --prefix=/usr/local --with-python=python3 && \
        ./b2 stage -j16 threading=multi link=shared cxxflags="-std=c++11" && \
        sudo ./b2 install threading=multi link=shared --with-system --with-filesystem && \
        ./b2 stage -j16 threading=multi link=static cxxflags="-std=c++11 -fpic" cflags="-fpic" && \
        sudo ./b2 install threading=multi link=static --with-system --with-filesystem && cd ../ && \
        git clone -b 2.0.6.2 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b 0.92  https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=OCL ../ && make -j4 && sudo make install && cd

WORKDIR /workspace

# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake ../MIVisionX && make -j8 && sudo make install && cd
