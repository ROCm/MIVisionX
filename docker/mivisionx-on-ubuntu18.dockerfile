FROM ubuntu:18.04

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN apt-get -y install gcc g++ cmake git
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN apt-get -y install libnuma-dev wget sudo gnupg2 kmod &&  \
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - && \
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list && \
        sudo apt-get update -y && \
        sudo apt-get -y install rocm-dev
# install OpenCV & FFMPEG - Level 3
RUN apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/3.4.0.zip && unzip 3.4.0.zip && \
        mkdir build && cd build && cmake -DWITH_OPENCL=OFF ../opencv-3.4.0 && make -j8 && sudo make install && sudo ldconfig && cd
RUN apt-get -y install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev \
        libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev \
        nasm yasm libx264-dev libx265-dev libnuma-dev libfdk-aac-dev && \
        git clone --recursive -b n4.0.4 https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg && sudo ldconfig && \
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/" && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && sudo make install && cd
# install MIVisionX neural net dependency - Level 4
RUN apt-get -y install sqlite3 libsqlite3-dev libbz2-dev libssl-dev python-dev python3-dev autoconf automake libtool curl make g++ unzip && \
        mkdir neuralNet && cd neuralNet && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo cp half-files/include/half.hpp /usr/local/include/ && \
        wget https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.bz2 && tar xjvf boost_1_72_0.tar.bz2 && \
        cd boost_1_72_0 && ./bootstrap.sh --prefix=/usr/local --with-python=python3 && \
        ./b2 stage -j16 threading=multi link=shared && sudo ./b2 install threading=multi link=shared --with-system --with-filesystem && \
        sudo ./b2 install threading=multi link=static --with-system --with-filesystem && cd ../ && \
        git clone https://github.com/RadeonOpenCompute/rocm-cmake.git && cd rocm-cmake && mkdir build && cd build && \
        cmake ../ && make -j8 && sudo make install && cd ../../ && \
        wget https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/archive/1.1.5.zip && unzip 1.1.5.zip && \
        cd MIOpenGEMM-1.1.5 && mkdir build && cd build && cmake ../ && make -j8 && sudo make install && cd ../../ && \
        wget https://github.com/ROCmSoftwarePlatform/MIOpen/archive/2.9.0.zip && unzip 2.9.0.zip && \
        cd MIOpen-2.9.0 && sudo cmake -P install_deps.cmake --minimum && mkdir build && cd build && \
        cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_USE_MIOPENGEMM=On ../ && make -j8 && make MIOpenDriver && sudo make install && cd ../../ && \
        git clone -b v3.12.0 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd
# install MIVisionX RALI dependency - Level 5
RUN apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev clang && \
        git clone -b 2.0.6.1 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd && \
        git clone -b 0.7  https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=OCL ../ && make -j4 && sudo make install && cd
# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake ../MIVisionX && make -j8 && sudo make install && cd