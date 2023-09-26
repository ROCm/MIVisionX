FROM ubuntu:20.04

ARG ROCM_INSTALLER_REPO=https://repo.radeon.com/amdgpu-install/5.6.1/ubuntu/focal/amdgpu-install_5.6.50601-1_all.deb
ARG ROCM_INSTALLER_PACKAGE=amdgpu-install_5.6.50601-1_all.deb

ENV MIVISIONX_DEPS_ROOT=/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git libcanberra-gtk-module
# install ROCm for mivisionx OpenCL/HIP dependency - Level 2
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration &&  \
        sudo apt-get -y clean && dpkg --add-architecture i386 && \
        wget ${ROCM_INSTALLER_REPO} && \
        sudo apt-get install -y ./${ROCM_INSTALLER_PACKAGE} && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=graphics,rocm
# RPP          
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install clang libboost-filesystem-dev half && \
        git clone -b 1.2.0 https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && \
        cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=HIP ../ && make -j4 && sudo make install
# install mivisionx package dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install half miopen-hip-dev rocblas-dev migraphx migraphx-dev 
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg libavcodec-dev libavformat-dev libswscale-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install libopencv-dev 
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install liblmdb-dev rapidjson-dev libturbojpeg0-dev libprotobuf-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3-dev python3-setuptools python3-pip python3-opencv pybind11-dev

ENV MIVISIONX_WORKSPACE=/workspace
WORKDIR $MIVISIONX_WORKSPACE

ENV PATH=$PATH:/opt/rocm/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

# Clone MIVisionX 
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        mkdir build && cd build && cmake -D BACKEND=HIP -D ROCAL=OFF ../MIVisionX && make -j8 && make install