FROM ubuntu:20.04

ARG ROCM_INSTALLER_REPO=https://repo.radeon.com/amdgpu-install/5.7/ubuntu/focal/amdgpu-install_5.7.50700-1_all.deb
ARG ROCM_INSTALLER_PACKAGE=amdgpu-install_5.7.50700-1_all.deb

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

# install mivisionx package dependencies
# VX_NN & VX_MIGraphX
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install half miopen-hip-dev rocblas-dev migraphx migraphx-dev
# VX_MEDIA
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg libavcodec-dev libavformat-dev libswscale-dev
# VX_openCV
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install libopencv-dev
# RPP          
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install clang half rpp
# rocAL
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install liblmdb-dev rapidjson-dev libturbojpeg0-dev libprotobuf-dev
# rocAL PyBind
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3-dev python3-setuptools python3-pip python3-opencv pybind11-dev

ENV MIVISIONX_WORKSPACE=/workspace
WORKDIR $MIVISIONX_WORKSPACE

ENV PATH=$PATH:/opt/rocm/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

# Clone MIVisionX 
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        mkdir build && cd build && cmake -D BACKEND=HIP -D ROCAL=OFF ../MIVisionX && make -j8 && make install