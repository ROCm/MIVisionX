FROM ubuntu:22.04

ENV MIVISIONX_DEPS_ROOT=/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git

# install ROCm for mivisionx OpenCL/HIP dependency - Level 2
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration &&  \
        wget https://repo.radeon.com/amdgpu-install/5.3/ubuntu/jammy/amdgpu-install_5.3.50300-1_all.deb && \
        sudo apt-get install -y ./amdgpu-install_5.3.50300-1_all.deb && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=graphics,rocm

# install OpenCV & FFMPEG - Level 3
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python3-dev python3-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/4.6.0.zip && unzip 4.6.0.zip && \
        mkdir build && cd build && cmake -DWITH_GTK=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_OPENCL=OFF ../opencv-4.6.0 && make -j8 && sudo make install && sudo ldconfig && cd
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev \
        libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev \
        nasm yasm libx264-dev libx265-dev libnuma-dev libfdk-aac-dev && \
        wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.4.2.zip && unzip n4.4.2.zip && cd FFmpeg-n4.4.2/ && sudo ldconfig && \
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/" && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && sudo make install && cd

# install MIVisionX neural net dependency - Level 4
RUN apt-get -y install rocblas rocblas-dev miopen-hip miopen-hip-dev migraphx

WORKDIR /workspace

# Clone MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        mkdir build && cd build && cmake -DBACKEND=HIP ../MIVisionX && make -j8 && make install