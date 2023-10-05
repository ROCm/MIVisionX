ARG TENSORFLOW_VERSION=latest
FROM rocm/tensorflow:${TENSORFLOW_VERSION}

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git
# install ROCm for mivisionx OpenCL/HIP dependency - Level 2
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration 
        #&&  \
        #wget https://repo.radeon.com/amdgpu-install/22.20/ubuntu/focal/amdgpu-install_22.20.50200-1_all.deb && \
        #sudo apt-get install -y ./amdgpu-install_22.20.50200-1_all.deb && \
        #sudo apt-get update -y && \
        #sudo amdgpu-install -y --usecase=graphics,rocm
# install OpenCV & FFMPEG - Level 3
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip && \
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
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install miopen-hip migraphx && \
        mkdir neuralNet && cd neuralNet && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo mkdir -p /usr/local/include/half && sudo cp half-files/include/half.hpp /usr/local/include/half && cd
# install MIVisionX rocAL dependency - Level 5
ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install wget libbz2-dev libssl-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev clang && \
        git clone -b 2.0.6.1 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b 1.2.0 https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=HIP ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b v3.12.4 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install git g++ hipblas hipsparse rocrand hipfft rocfft rocthrust-dev hipcub-dev python3 python3-pip python3-dev && \
        git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && mkdir build && cd build && \
        cmake ../ && make -j4 && sudo make install && cd ../../ && \
        pip install pytest==3.1 && git clone -b v2.10.4 https://github.com/pybind/pybind11 && cd pybind11 && mkdir build && cd build && \
        cmake -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b v12.2.0 https://github.com/ROCmSoftwarePlatform/cupy.git && cd cupy && git submodule update --init && \
        pip install -e . --no-cache-dir -vvvv && pip install numpy==1.21 && cd

WORKDIR /workspace

# Clone MIVisionX 
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
        #mkdir build && cd build && cmake -DBACKEND=HIP ../MIVisionX && make -j8 && make install
#install tensorflow dependencies - Compile protos and Install TensorFlow Object Detection API.
RUN cd ~/models/research/ && \
protoc object_detection/protos/*.proto --python_out=. && \
cp object_detection/packages/tf2/setup.py . && \
python -m pip install .

#upgrade pip
RUN pip3 install --upgrade pip

